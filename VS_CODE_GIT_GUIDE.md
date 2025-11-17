# VS Code Git 集成完整指南

这份指南将帮助你充分利用 VS Code 的 Git 集成功能来管理 RoGs 项目。

## 基础 Git 面板使用

### 打开 Git 面板
- **快捷键**: `Ctrl + Shift + G`
- **菜单**: 查看 → 源代码管理
- **图标**: 左侧活动栏的 Git 图标

### Git 面板结构
```
源代码管理 (Ctrl+Shift+G)
├── 更改 (Modified files)
├── 暂存的更改 (Staged files)
├── 提交消息输入框
├── 提交按钮 (✓)
└── 更多操作 (...) 菜单
```

## 文件状态图标说明

在 VS Code 中，文件有不同的状态图标：

- **U** (Untracked): 未跟踪的新文件
- **M** (Modified): 已修改的文件
- **A** (Added): 已添加到暂存区的文件
- **D** (Deleted): 已删除的文件
- **R** (Renamed): 重命名的文件

## 日常操作步骤

### 1. 查看更改
1. 打开 Git 面板 (`Ctrl+Shift+G`)
2. 在"更改"部分查看所有修改的文件
3. 点击文件名查看具体的 diff

### 2. 暂存文件 (Stage)
**方法一**: 点击文件右侧的 `+` 按钮
**方法二**: 右键点击文件 → "暂存更改"
**方法三**: 批量暂存 - 点击"更改"标题旁的 `+` 按钮

### 3. 取消暂存 (Unstage)
**方法一**: 在"暂存的更改"中点击文件右侧的 `-` 按钮
**方法二**: 右键点击文件 → "取消暂存更改"

### 4. 提交更改
1. 在消息框中输入提交信息
2. 按 `Ctrl+Enter` 或点击 ✓ 按钮
3. 也可以点击"提交"按钮下拉菜单选择"提交和推送"

## 高级功能

### 时间线视图 (Timeline)
1. 打开任意文件
2. 右键点击编辑器 → "时间线" 或
3. 使用命令面板 (`Ctrl+Shift+P`) → "时间线: 显示时间线"
4. 查看文件的历史版本和提交记录

### 分支管理
1. 点击左下角的当前分支名称
2. 或者使用命令面板 → "Git: 检出到..."
3. 可以：
   - 创建新分支
   - 切换分支
   - 删除分支
   - 合并分支

### 合并冲突解决
当发生合并冲突时：
1. VS Code 会高亮显示冲突区域
2. 提供三个选项：
   - 接受当前更改 (Accept Current)
   - 接受传入更改 (Accept Incoming)
   - 接受双方更改 (Accept Both)
   - 比较变更 (Compare Changes)
3. 手动编辑解决冲突后保存文件
4. 暂存解决冲突的文件
5. 完成合并提交

## 推荐插件

### GitLens (强烈推荐)
```bash
code --install-extension eamodio.gitlens
```

功能：
- **行级 blame**: 显示每行代码的最后修改者和时间
- **提交详情**: 悬停查看提交详细信息
- **文件历史**: 浏览文件历史版本
- **分支比较**: 比较不同分支的差异
- **Git 命令面板**: 快速访问 Git 命令

### Git Graph
```bash
code --install-extension mhutchie.git-graph
```
功能：
- 可视化的分支图
- 交互式的分支和提交管理
- 轻松创建、切换、合并分支

### GitHub Pull Requests and Issues
```bash
code --install-extension GitHub.vscode-pull-request-github
```
功能：
- 在 VS Code 内查看和管理 PR
- 审查代码和添加评论
- 管理 GitHub Issues

## 快捷键大全

### 基本操作
- `Ctrl+Shift+G`: 打开 Git 面板
- `Ctrl+Enter`: 提交更改
- `Ctrl+Shift+P` → "Git: ": Git 命令面板

### 文件操作
- 在 Git 面板中：
  - `Enter`: 查看文件 diff
  - `Space`: 暂存/取消暂存文件
  - `Delete`: 放弃文件更改

### 编辑器内操作
- 右键点击行号 → "Git 镜头" 查看该行的 Git 信息
- `Alt+F3`: 查找下一个冲突

## 实际工作流程示例

### 场景 1: 添加新功能
1. **创建功能分支**:
   - 点击左下角分支名称
   - 选择"创建新分支"
   - 输入分支名: `feature/new-kdtree-optimization`

2. **编写代码**:
   - 修改代码文件
   - 观察 Git 面板中的更改

3. **暂存和提交**:
   ```
   1. 在 Git 面板查看更改
   2. 点击 + 暂存需要的文件
   3. 输入提交信息: "feat: add new KD-tree optimization algorithm"
   4. Ctrl+Enter 提交
   ```

4. **推送到远程**:
   - 点击"..."菜单 → "推送"
   - 或使用命令面板 → "Git: 推送"

### 场景 2: 代码审查
1. **查看更改**:
   - 在 Git 面板点击每个文件查看 diff
   - 使用 GitLens 查看每行的修改历史

2. **回退更改**:
   - 对于不需要的更改，点击垃圾桶图标放弃
   - 对于已提交的更改，使用 GitLens 的"还原提交"功能

### 场景 3: 解决合并冲突
1. **发现冲突**:
   - VS Code 会自动高亮冲突文件
   - 文件图标显示为红色

2. **解决冲突**:
   - 打开冲突文件
   - 使用"接受当前/传入/双方"按钮
   - 或手动编辑代码

3. **完成合并**:
   - 暂存解决冲突的文件
   - 提交合并

## 自定义设置

### settings.json 配置
```json
{
    "git.enableSmartCommit": true,
    "git.confirmSync": false,
    "git.autofetch": true,
    "git.enableStatusBarSync": true,
    "gitlens.hovers.enabled": true,
    "gitlens.codeLens.enabled": true,
    "gitlens.currentLine.enabled": true
}
```

### 有用的设置说明
- `git.enableSmartCommit`: 自动暂存所有更改后提交
- `git.autofetch`: 自动获取远程更改
- `git.confirmSync`: 同步时不需要确认

## 故障排除

### 常见问题

1. **Git 面板空白**:
   - 检查是否在 Git 仓库内
   - 重新加载窗口 (`Ctrl+R`)

2. **无法推送**:
   - 检查网络连接
   - 验证远程仓库配置
   - 检查认证设置

3. **GitLens 不显示**:
   - 确保文件在 Git 仓库内
   - 检查 GitLens 设置
   - 重新加载窗口

### 调试技巧
- 使用 Git 输出面板查看详细日志
- 命令面板 → "Git: 显示 Git 输出"
- 检查状态栏的 Git 状态指示器

## 最佳实践总结

1. **频繁提交**: 小步快跑，频繁提交
2. **有意义的提交信息**: 清晰描述更改内容
3. **使用分支**: 每个功能使用独立分支
4. **及时推送**: 定期推送到远程仓库
5. **利用可视化工具**: 使用 VS Code 的图形界面简化操作
6. **安装推荐插件**: GitLens 等插件能大幅提升效率

记住：VS Code 的 Git 集成非常强大，多用图形界面可以减少命令行操作，提高开发效率！