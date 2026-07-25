本文档总结了3GPP RAG工程在GB10服务器上部署后，远程电脑访问前端时所有按钮失效的问题排查与修复全过程。

标签：`#troubleshooting` `#前端` `#NextJS` `#远程访问` `#GB10`

## 1 问题现象
在GB10服务器上启动前后端服务后，通过服务器IP远程访问前端页面时出现以下现象：
- 页面可以正常加载（HTTP 200），HTML结构完整可见
- 所有交互按钮完全失效，包含明暗模式切换、历史记录、发送等纯前端交互功能
- 在服务器本机使用`localhost`访问时，页面所有功能运行正常

## 2 根因分析
### 2.1 直接原因：React Hydration（水合）失败
Next.js 16开发模式（`next dev`）依靠HMR（热模块替换）WebSocket连接来驱动React水合流程。
当远程浏览器访问开发环境服务时，若请求来源Origin不在`allowedDevOrigins`允许列表内，HMR对应的WebSocket连接就会被拒绝，React无法完成客户端水合，页面退化为静态纯HTML，所有绑定JS事件的交互按钮都会失去响应。

### 2.2 深层原因：`allowedDevOrigins` 不支持通配符
Next.js 16新增`allowedDevOrigins`安全配置项，用来限定可接入开发服务的请求来源，关键特性如下：
- 配置`allowedDevOrigins: ["*"]`不会生效，Next.js16不支持通配符`*`，该配置会被引擎直接忽略
- 该字段必须填写具体的主机名、IP地址
- 未配置或者配置无效时，Next.js会在运行日志输出告警，示例配置提示：
```js
To allow this host in development, add it to "allowedDevOrigins" in next.config.js:
module.exports = {
  allowedDevOrigins: ['109.105.35.139'],
}
```

### 2.3 完整问题链路
```
远程浏览器访问 http://<服务器IP>:3000
→ Next.js dev服务校验请求Origin
→ Origin不在`allowedDevOrigins`内（通配符*配置被忽略）
→ HMR WebSocket连接被拒绝
→ React Hydration 水合失败
→ 页面退化为无JS交互能力的纯HTML
→ 全部按钮交互失效
```

## 3 修复方案
### 3.1 自动获取本机IP，注入`allowedDevOrigins`
修改`frontend/next.config.ts`，借助Node.js内置`os`模块的`networkInterfaces()`接口，自动读取本机所有网卡的IPv4地址，动态加入`allowedDevOrigins`列表，代码实现：
```ts
import type { NextConfig } from "next";
import os from "os";

// 自动获取本机所有非内网IPv4地址，加入allowedDevOrigins
function getAllLocalIPs(): string[] {
  const ips: string[] = [];
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name] || []) {
      if (iface.family === "IPv4" && !iface.internal) {
        ips.push(iface.address);
      }
    }
  }
  return ips;
}

const autoIPs = getAllLocalIPs();
const extraOrigins = process.env.DEV_ORIGIN
  ? process.env.DEV_ORIGIN.split(",").map(s => s.trim()).filter(Boolean)
  : [];
const devOrigins = [...autoIPs, ...extraOrigins];

const nextConfig: NextConfig = {
  allowedDevOrigins: devOrigins,
  // 其余原有配置
};
```
本方案优势：
- 部署到任意服务器，无需手动填写IP到配置文件
- 天然适配多网卡服务器环境
- 可通过`DEV_ORIGIN`环境变量追加额外允许的访问源地址

### 3.2 搭配rewrites反向代理，消除前端硬编码后端IP
同步调整请求链路，前端API改用相对路径，借助Next.js的rewrites能力做后端反向代理，规避跨域、前端写死后端IP的问题。
`frontend/next.config.ts`新增rewrites配置：
```ts
async rewrites() {
  return [
    {
      source: "/api/v1/:path*",
      destination: `${BACKEND_URL}/api/v1/:path*`,
    },
  ];
},
```
修改`frontend/lib/api.ts`，把API基础地址改为相对路径：
```ts
const API_URL = "/api/v1"; // 请求交由Next.js服务端转发至后端
```
该优化的优势：
- 前端统一使用相对路径发起请求，浏览器请求目标和前端站点同源
- 请求经由Next.js服务端转发到后端，彻底规避浏览器跨域限制
- 服务迁移部署时，前端无需修改后端IP，仅调整环境变量里的`BACKEND_URL`即可

## 4 验证方法
### 4.1 校验`allowedDevOrigins`告警是否消除
执行如下shell命令检索前端日志：
```bash
grep -c "allowedDevOrigins" logs/frontend.log
```
命令输出结果为`0`，代表对应告警已经消除，配置生效。

### 4.2 接口层面远程访问校验
```bash
# 校验前端主页远程访问
curl -s -o /dev/null -w "HTTP %{http_code}" http://<服务器IP>:3000
# 校验管理员登录页远程访问
curl -s -o /dev/null -w "HTTP %{http_code}" http://<服务器IP>:3000/admin/login
# 校验前端代理转发后端健康检查接口
curl -s http://<服务器IP>:3000/api/v1/health
```

### 4.3 浏览器端功能验证
其他终端浏览器访问`http://<服务器IP>:3000`，使用快捷键`Ctrl+Shift+R`强制刷新清空缓存后，核对功能：
- ✅ 明暗模式切换按钮可正常点击
- ✅ 历史记录按钮可正常点击
- ✅ 发送按钮可正常点击
> 备注：LLM相关接口未配置密钥时，发送请求会返回`Connection error`，属于预期正常现象，不影响前端按钮交互能力。

## 5 涉及修改的相关文件
| 文件路径 | 修改内容 |
| ---- | ---- |
| `frontend/next.config.ts` | 1.新增自动获取本机IP逻辑，动态填充`allowedDevOrigins`；2.新增rewrites反向代理配置 |
| `frontend/lib/api.ts` | 将硬编码的API基础URL修改为相对路径`/api/v1` |
| `frontend/.env.local` | 删除`NEXT_PUBLIC_API_URL`配置项，仅保留`BACKEND_URL`（默认指向本机后端） |

## 6 注意事项
### 6.1 生产环境无需该类开发配置
`allowedDevOrigins`仅对`next dev`开发模式生效；生产环境执行`next build && next start`启动服务时，不存在该访问源校验限制，无需配置该项。

### 6.2 LLM接口未配置的表现说明
若环境变量内`LLM_API_KEY`仅为占位值、没有填入正式密钥，问答功能调用时会返回`Connection error`，属于正常情况；检索功能不依赖LLM，可以正常使用。后续拿到正式LLM接入凭证后，更新`.env`内`LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`三个参数，即可启用完整问答能力。

### 6.3 浏览器缓存清理要求
前端配置更新重启服务后，远程访问的浏览器必须执行`Ctrl+Shift+R`强制刷新，清除旧的JS资源缓存；否则浏览器仍会加载内嵌`localhost`地址的旧脚本，导致问题复现。

> 原始记录信息：Xiaoyang Li|xiaoyang.li|vRAN Solution Lab|2026-07-17 20:23:36