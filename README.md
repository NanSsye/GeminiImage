# 🎨 Gemini 图像生成器 (GeminiImage)

> 🚀 使用 Google 最先进的 Gemini AI 模型生成和编辑精美图像！
> **本插件是 [XYBotv2](https://github.com/HenryXiaoYang/XYBotv2) 的一个插件。**

<img src="https://github.com/user-attachments/assets/a2627960-69d8-400d-903c-309dbeadf125" width="400" height="600">

## ✨ 功能特点

- 🖼️ **AI 图像生成** - 从文本描述生成高质量图像
- 🎭 **图像编辑** - 修改现有图像，添加新元素或改变风格
- 💬 **上下文对话** - 支持多轮对话，连续优化你的创意
- 🔄 **自动使用上一张图片** - 多轮对话时自动将上一次生成的图片作为下一次编辑的输入
- 📝 **显示 AI 文本解释** - 展示模型对生成图片的文字解释，帮助理解创作过程
- 🌐 **代理支持** - 支持 HTTP/SOCKS 代理，解决网络访问问题
- 🚪 **手动结束对话** - 支持随时结束对话会话，提高资源利用效率
- 🧠 **Gemini 2.0** - 使用 Google 最新的 Gemini 2.0 Flash 实验版模型
- 🔄 **简单易用** - 简洁的命令，快速获得结果

## 📋 使用指南

### 图像生成命令

使用以下命令生成图像：

```
#生成图片 [你的描述]
#画图 [你的描述]
#图片生成 [你的描述]
```

### 图像编辑命令

上传图片并添加以下描述：

```
#编辑图片 [编辑描述]
#修改图片 [编辑描述]
```

### 结束对话命令

使用以下任一命令结束当前对话：

```
#结束对话
#退出对话
#关闭对话
#结束
```

### 🔄 连续对话

启动对话后，可以直接发送新消息调整或修改，无需重复命令前缀。系统会自动使用上一次生成的图片作为输入，例如：

1. 首先发送：`#生成图片 一只猫咪`
2. 然后直接发送：`给它戴上一顶帽子`（无需命令前缀）
3. 再发送：`背景改成草地`（会基于上一张图片继续修改）
4. 最后发送：`#结束对话`（结束当前会话，释放资源）

## 💎 示例提示词

### 生成图像

- `#生成图片 一只戴着太阳镜的猫咪在沙滩上冲浪`
- `#画图 未来城市中的飞行汽车和高耸入云的建筑`
- `#图片生成 充满星星的夜空下的湖泊，倒映着北极光`

### 编辑图像

- `#编辑图片 将背景改为繁华的城市街道`
- `#修改图片 给人物添加一顶帽子和一副墨镜`

## ⚙️ 配置说明

在`config.toml`中设置：

```toml
[GeminiImage]
# 基本配置
enable = true
gemini_api_key = "你的API密钥"
model = "gemini-2.0-flash-exp-image-generation"

# 命令配置
commands = ["#生成图片", "#画图", "#图片生成"]
edit_commands = ["#编辑图片", "#修改图片"]

# 积分配置
enable_points = true
generate_image_cost = 10
edit_image_cost = 15

# 图片保存路径
save_path = "temp"

# 管理员列表(免费使用)
admins = []

# 代理配置
enable_proxy = false                  # 是否启用代理
proxy_url = "http://127.0.0.1:7890"   # 代理服务器地址
```

### 代理设置说明

如果您无法直接访问 Google Gemini API，可以配置代理：

1. 将`enable_proxy`设置为`true`
2. 在`proxy_url`中设置您的代理服务器地址：
   - HTTP 代理格式：`http://127.0.0.1:7890`
   - SOCKS5 代理格式：`socks5://127.0.0.1:1080`

注意：使用代理前请确保代理服务器正常工作且能访问 Google 服务。

## 📊 积分系统

- 生成图片：消耗 10 积分
- 编辑图片：消耗 15 积分
- 管理员可免费使用

## 📝 开发日志

- v1.0.0: 初始版本发布
- v1.1.0: 添加上下文对话支持
- v1.2.0: 添加文本响应显示和自动使用上一张图片功能
- v1.3.0: 添加代理支持，解决网络访问问题
- v1.4.0: 添加手动结束对话功能

## 👨‍💻 作者

**老夏的金库** ©️ 2024

**给个 ⭐ Star 支持吧！** 😊

**开源不易，感谢打赏支持！**

![image](https://github.com/user-attachments/assets/2dde3b46-85a1-4f22-8a54-3928ef59b85f)

## �� 许可证

MIT License
