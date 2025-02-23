# 三巨头辩论系统 (ThreeAIHeads)

一个基于 LangChain 的多模型辩论系统，让多个顶级 AI 模型为您的问题展开专业辩论并达成共识。

## 📖 特性

- 🤖 支持多个AI模型同时参与辩论
- 🎯 自动分配supervisor和普通辩论者角色
- 📊 实时展示每个模型的观点和响应时间
- 🤝 自动生成最终共识建议
- 🚀 简洁的 Gradio Web 界面

## 🛠 安装

1. 克隆项目
```bash
git clone https://github.com/mcp2everything/ThreeAIHeads.git
cd ThreeAIHeads
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
复制 `.env.example` 为 `.env` 并填入你的API密钥：
```properties
MODEL1_NAME=your-model1-name
MODEL1_BASE_URL=your-model1-base-url
MODEL1_API_KEY=your-model1-api-key

MODEL2_NAME=your-model2-name
MODEL2_BASE_URL=your-model2-base-url
MODEL2_API_KEY=your-model2-api-key

MODEL3_NAME=your-model3-name
MODEL3_BASE_URL=your-model3-base-url
MODEL3_API_KEY=your-model3-api-key
```

## 🚀 使用方法

1. 启动系统
```bash
python app.py
```

2. 打开浏览器访问显示的地址（通常是 http://127.0.0.1:7860）

3. 在输入框中输入你的问题，例如：
   - "西门子温控系统当前采样周期为2秒，是否值得改为4秒？"
   - "在工业生产环境中，如何权衡实时性和系统负载？"

4. 点击"开始辩论"按钮，等待结果

## 🌟 实际效果

系统会按以下步骤工作：
1. 第一个模型作为supervisor提供初始观点
2. 其他模型基于已有观点提供补充或反驳
3. 最后生成一份综合各方观点的最终共识

输出包含：
- 每个模型的详细观点
- 最终达成的共识建议
- 具体的执行建议

## 🔧 故障排除

- 如果模型响应超时，检查网络连接和API配置
- 确保所有环境变量正确设置
- 查看控制台输出的详细错误信息

## 📝 许可证

MIT License

## 🤝 贡献

欢迎提交 Issues 和 Pull Requests！

## 📞 联系

- GitHub: [@mcp2everything](https://github.com/mcp2everything)
- Project: [ThreeAIHeads](https://github.com/mcp2everything/ThreeAIHeads)