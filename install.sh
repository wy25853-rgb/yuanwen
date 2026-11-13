#!/usr/bin/env bash
# 原文库 安装/更新脚本。重复运行是安全的：不会删除已有数据，也不会更换连接地址。
# 域名、连接密钥、称呼都只保存在服务器本机的 /opt/yuanwen 里，不在这个仓库中。
set -euo pipefail

DIR="/opt/yuanwen"
RAW="https://raw.githubusercontent.com/wy25853-rgb/yuanwen/main"

say() { echo -e "\n\033[1;32m>>> $*\033[0m"; }
die() { echo -e "\n\033[1;31m出错了：$*\033[0m\n请把上面的内容截图发给 Claude。"; exit 1; }

[ "$(id -u)" = "0" ] || die "请用 root 用户运行"
command -v docker >/dev/null || die "没有找到 Docker"

# 域名：优先用环境变量，其次沿用服务器上已有的配置
if [ -z "${DOMAIN:-}" ] && [ -f "$DIR/Caddyfile" ]; then
  DOMAIN="$(head -1 "$DIR/Caddyfile" | awk '{print $1}')"
fi
[ -n "${DOMAIN:-}" ] || die "不知道域名。第一次安装请这样运行：DOMAIN=你的域名 bash install.sh"

say "1/6 下载最新程序到 $DIR"
mkdir -p "$DIR/data" "$DIR/caddy_data" "$DIR/src.new"
for f in app.py Dockerfile requirements.txt; do
  curl -fsSL "$RAW/$f?t=$(date +%s)" -o "$DIR/src.new/$f" || die "下载 $f 失败，可能是网络问题，过一会儿再试"
done
grep -q "MCPServer" "$DIR/src.new/app.py" || die "下载到的程序不完整"
rm -rf "$DIR/src" && mv "$DIR/src.new" "$DIR/src"

say "2/6 生成连接地址里的随机密钥（已有则沿用）"
if [ ! -f "$DIR/.env" ]; then
  echo "SECRET_PATH=$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')" > "$DIR/.env"
  chmod 600 "$DIR/.env"
fi
# 称呼：运行时用 USER_NAME=… AI_NAME=… 传入则写入（或更新）服务器本机配置
for k in USER_NAME AI_NAME; do
  v="$(eval "echo \"\${$k:-}\"")"
  if [ -n "$v" ]; then
    sed -i "/^$k=/d" "$DIR/.env"; echo "$k=$v" >> "$DIR/.env"
  fi
done
SECRET="$(grep '^SECRET_PATH=' "$DIR/.env" | cut -d= -f2)"
[ ${#SECRET} -ge 16 ] || die "密钥文件 $DIR/.env 不正常"

cat > "$DIR/Caddyfile" <<CADDY
$DOMAIN {
    encode gzip
    reverse_proxy yuanwen:8000
}
CADDY

say "3/6 检查 80 和 443 端口"
for p in 80 443; do
  if ss -tln | awk '{print $4}' | grep -qE "[:.]$p\$"; then
    docker ps --format '{{.Names}}' | grep -qx yuanwen-caddy || die "端口 $p 已被别的程序占用"
  fi
done
if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  ufw allow 80/tcp >/dev/null; ufw allow 443/tcp >/dev/null; echo "已在防火墙放行 80 和 443"
fi

say "4/6 构建程序（第一次要几分钟，请耐心等）"
docker build -q -t yuanwen "$DIR/src" >/dev/null || die "构建失败"

say "5/6 启动服务"
docker network inspect yuanwen-net >/dev/null 2>&1 || docker network create yuanwen-net >/dev/null
docker rm -f yuanwen yuanwen-caddy >/dev/null 2>&1 || true
docker run -d --name yuanwen --restart unless-stopped --network yuanwen-net \
  --env-file "$DIR/.env" -v "$DIR/data:/data" yuanwen >/dev/null
docker run -d --name yuanwen-caddy --restart unless-stopped --network yuanwen-net \
  -p 80:80 -p 443:443 -v "$DIR/Caddyfile:/etc/caddy/Caddyfile:ro" \
  -v "$DIR/caddy_data:/data" caddy:2 >/dev/null

say "6/6 检查是否正常（最多等 2 分钟）"
ok=""
for i in $(seq 1 24); do
  code="$(curl -s -o /dev/null -m 8 -w '%{http_code}' "https://$DOMAIN/healthz" || true)"
  [ "$code" = "200" ] && { ok=1; break; }
  sleep 5
done

cat > /usr/local/bin/yuanwen-update <<UPD
#!/usr/bin/env bash
curl -fsSL "$RAW/install.sh?t=\$(date +%s)" | bash
UPD
chmod +x /usr/local/bin/yuanwen-update

cat > "$DIR/address.txt" <<TXT
Claude 连接器地址： https://$DOMAIN/$SECRET/mcp
上传导出包的页面： https://$DOMAIN/$SECRET/upload
TXT
chmod 600 "$DIR/address.txt"

if [ -n "$ok" ]; then
  echo -e "\n\033[1;32m安装成功，已是最新版。\033[0m"
else
  echo -e "\n\033[1;33m服务已启动，但从外面访问 https://$DOMAIN 还没通。\033[0m"
  echo "---- 程序日志 ----";  docker logs --tail 15 yuanwen 2>&1 || true
  echo "---- Caddy 日志 ----"; docker logs --tail 15 yuanwen-caddy 2>&1 | cut -c1-300 || true
  echo "请把这一屏截图发给 Claude（下面两行地址请遮住）。"
fi
echo
cat "$DIR/address.txt"
echo
echo "这两个地址相当于钥匙，不要发给别人。以后忘了，运行：cat $DIR/address.txt"
echo "以后更新程序，只需运行：yuanwen-update"
exit 0
