"""
remote.py —— AutoDL 服务器连接工具（基于 paramiko）

读取 server.conf（host / port / user / password / key_file）：
  1. 先用密钥登录，失败再用密码登录
  2. setup-key 把本地公钥写进服务器 ~/.ssh/authorized_keys，之后永久免密
  3. exec 远程执行命令；put 上传文件（支持断点续传，Windows 没有 rsync）

用法：
    python remote.py test                          # 测试连接
    python remote.py setup-key                     # 配置免密（只需跑一次）
    python remote.py exec "nvidia-smi"             # 执行远程命令
    python remote.py exec "df -h /root/autodl-tmp" --stream
    python remote.py put smolvla_base.tar /root/autodl-tmp/   # 上传（可中断后重跑）
"""

import os
import sys
import time

import paramiko

ROOT = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.join(ROOT, "server.conf")


def load_conf():
    conf = {}
    with open(CONF, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            conf[k.strip()] = v.strip().strip('"').strip("'")
    conf["port"] = int(conf.get("port", 22))
    return conf


def read_pubkey(key_file):
    pub = key_file + ".pub"
    if not os.path.isfile(pub):
        return None
    with open(pub, encoding="utf-8") as f:
        return f.read().strip()


def connect(conf, allow_password=True):
    """优先密钥登录，失败再退回密码。"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    key_file = conf.get("key_file")
    if key_file and os.path.isfile(key_file):
        try:
            client.connect(
                hostname=conf["host"], port=conf["port"], username=conf["user"],
                key_filename=key_file, timeout=30, banner_timeout=30,
            )
            return client
        except Exception as e:  # noqa: BLE001
            print(f"[info] 密钥登录失败（{type(e).__name__}），改用密码")

    if not allow_password:
        raise SystemExit("密钥登录失败，且未配置 password")

    pwd = conf.get("password", "")
    if not pwd or "在这里填密码" in pwd:
        raise SystemExit("server.conf 里还没填 password（或先跑 setup-key 配免密）")

    client.connect(
        hostname=conf["host"], port=conf["port"], username=conf["user"],
        password=pwd, timeout=30, banner_timeout=30,
        allow_agent=False, look_for_keys=False,
    )
    return client


def cmd_setup_key(conf):
    pub = read_pubkey(conf.get("key_file", ""))
    if not pub:
        raise SystemExit(f"找不到公钥：{conf.get('key_file')}.pub")

    ssh = connect(conf)
    sftp = ssh.open_sftp()
    d = "/root/.ssh"
    ak = f"{d}/authorized_keys"
    try:
        sftp.stat(d)
    except IOError:
        ssh.exec_command(f"mkdir -p {d} && chmod 700 {d}")[1].channel.recv_exit_status()

    existing = ""
    try:
        with sftp.open(ak, "r") as f:
            existing = f.read().decode("utf-8", "replace")
    except IOError:
        pass

    if pub in existing:
        print("[ok] 公钥已在服务器上，无需重复添加")
    else:
        prefix = "" if (existing.endswith("\n") or not existing) else "\n"
        with sftp.open(ak, "a") as f:
            f.write(prefix + pub + "\n")
        sftp.chmod(ak, 0o600)
        print(f"[ok] 公钥已写入 {ak}")

    sftp.close()
    ssh.close()

    # 验证：只允许密钥登录
    ssh2 = connect(conf, allow_password=False)
    out, _err, _code = run(ssh2, "hostname && echo KEY_OK")
    print(f"[ok] 免密登录验证通过：{out.strip()}")
    ssh2.close()


def run(ssh, command, timeout=3600):
    _in, out, err = ssh.exec_command(command, timeout=timeout)
    code = out.channel.recv_exit_status()
    return out.read().decode("utf-8", "replace"), err.read().decode("utf-8", "replace"), code


def cmd_exec(conf, argv):
    stream = "--stream" in argv
    argv = [a for a in argv if a != "--stream"]
    ssh = connect(conf)
    command = " ".join(argv)
    if stream:
        _in, out, err = ssh.exec_command(command, get_pty=False)
        for line in out:
            print(line, end="")
        code = out.channel.recv_exit_status()
        sys.stderr.write(err.read().decode("utf-8", "replace"))
    else:
        o, e, code = run(ssh, command)
        if o:
            print(o, end="")
        if e:
            sys.stderr.write(e)
    ssh.close()
    return code


def cmd_put(conf, argv):
    if len(argv) < 2:
        raise SystemExit("用法: python remote.py put <本地文件> <远程目录或路径>")
    local, remote = argv[0], argv[1]
    if not os.path.isfile(local):
        raise SystemExit(f"本地文件不存在: {local}")

    ssh = connect(conf)
    sftp = ssh.open_sftp()
    if remote.endswith("/"):
        try:
            sftp.stat(remote)
            isdir = True
        except IOError:
            isdir = False
        remote = os.path.join(remote, os.path.basename(local)) if isdir else remote

    total = os.path.getsize(local)
    done = 0
    try:
        done = sftp.stat(remote).st_size
    except IOError:
        pass
    if done >= total and total > 0:
        print(f"[skip] {remote} 已存在且大小一致 ({total/2**30:.2f} GB)")
        sftp.close()
        ssh.close()
        return 0
    if done:
        print(f"[resume] 断点续传，从 {done/2**30:.2f} GB 继续")

    t0 = time.time()
    with open(local, "rb") as fl:
        fl.seek(done)
        with sftp.open(remote, "ab") as fr:
            while True:
                chunk = fl.read(4 << 20)
                if not chunk:
                    break
                fr.write(chunk)
                done += len(chunk)
                sp = done / max(time.time() - t0, 1e-6) / 2**20
                print(f"\r  {done/2**30:6.2f}/{total/2**30:.2f} GB  {sp:6.1f} MB/s", end="", flush=True)
    print()
    sftp.chmod(remote, 0o644)
    sftp.close()
    ssh.close()
    print(f"[ok] {os.path.basename(local)} -> {remote}")
    return 0


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    conf = load_conf()
    sub = argv[0]
    rest = argv[1:]

    if sub == "test":
        ssh = connect(conf)
        o, _e, _c = run(ssh, "hostname; whoami; uname -m; df -h /root/autodl-tmp | tail -1")
        print(o, end="")
        ssh.close()
        return 0
    if sub == "setup-key":
        cmd_setup_key(conf)
        return 0
    if sub == "exec":
        return cmd_exec(conf, rest)
    if sub == "put":
        return cmd_put(conf, rest)

    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
