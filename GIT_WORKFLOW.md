# 个人 Git 工作流指南

这份指南将帮助你管理 RoGs 项目的个人开发工作流。

## 当前设置状态

✅ **已完成设置：**
- 远程仓库已配置：
  - `origin` -> https://github.com/tanrenjun/RoGS.git (你的 fork)
  - `upstream` -> https://github.com/fzhiheng/RoGS.git (原始仓库)
- 当前在 `main` 分支，有未提交的修改

## 日常 Git 工作流

### 1. 开始工作前的日常更新
```bash
# 获取上游最新更改
git fetch upstream

# 查看上游是否有更新
git log HEAD..upstream/main --oneline

# 如果有更新，合并到本地
git merge upstream/main
```

### 2. 创建功能分支
```bash
# 为每个新功能创建独立分支
git checkout -b feature/your-feature-name

# 或者用于修复
git checkout -b fix/issue-description
```

### 3. 提交更改
```bash
# 查看当前修改
git status

# 添加特定文件到暂存区
git add filename

# 添加所有修改的文件
git add .

# 提交更改
git commit -m "描述你的更改"

# 推送到你的 fork
git push origin feature/your-feature-name
```

### 4. 分支管理策略

#### 主要分支
- `main`: 主分支，保持稳定
- `develop`: 开发分支（可选）
- `feature/*`: 功能开发分支
- `fix/*`: 错误修复分支
- `experiment/*`: 实验性分支

#### 推荐的工作流程
1. 从最新的 `main` 创建功能分支
2. 在功能分支上开发和提交
3. 完成后合并回 `main`
4. 推送到你的 fork

### 5. 实用 Git 命令

#### 查看历史和状态
```bash
# 查看提交历史
git log --oneline --graph --all

# 查看文件修改历史
git log --follow filename

# 查看具体修改内容
git diff

# 查看暂存区的修改
git diff --staged
```

#### 撤销和重置
```bash
# 撤销工作区的修改（单个文件）
git checkout -- filename

# 撤销暂存区的文件
git reset HEAD filename

# 撤销最后一次提交（保留修改）
git reset --soft HEAD~1

# 撤销最后一次提交（丢弃修改）
git reset --hard HEAD~1
```

#### 分支操作
```bash
# 列出所有分支
git branch -a

# 切换到现有分支
git checkout branch-name

# 删除本地分支
git branch -d branch-name

# 强制删除本地分支
git branch -D branch-name

# 删除远程分支
git push origin --delete branch-name
```

#### 同步和合并
```bash
# 从上游同步
git fetch upstream
git checkout main
git merge upstream/main

# 解决合并冲突后
git add conflicted-file
git commit
```

## VS Code Git 集成

### 1. 源代码管理面板 (Ctrl+Shift+G)
- **更改**: 查看所有修改的文件
- **暂存的更改**: 已准备提交的文件
- **提交**: 输入提交信息并提交

### 2. 时间线视图
- 右键点击文件 → "时间线" → 查看文件历史
- 可以比较不同版本的差异

### 3. GitLens 扩展（推荐）
```bash
# 安装 GitLens 扩展
code --install-extension eamodio.gitlens
```

功能：
- 行级 git  blame 信息
- 提交历史可视化
- 分支比较
- 文件历史浏览器

### 4. 快捷键
- `Ctrl+Shift+G`: 打开 Git 面板
- `Ctrl+Enter`: 提交更改
- `Ctrl+Shift+P` → "Git: " 前缀命令

## 推荐的开发工作流

### 每日工作流程
1. **早上开始工作时：**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   git push origin main
   ```

2. **创建当日工作分支：**
   ```bash
   git checkout -b feature/todays-work
   ```

3. **工作过程中定期提交：**
   ```bash
   git add .
   git commit -m "描述更改"
   ```

4. **一天结束时：**
   ```bash
   git push origin feature/todays-work
   ```

### 完整功能开发流程
1. **规划阶段：**
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/new-kdtree-optimization
   ```

2. **开发阶段：**
   ```bash
   # 编写代码...
   git add .
   git commit -m "feat: implement new KD-tree optimization"

   # 继续开发...
   git add .
   git commit -m "feat: add performance benchmarking"
   ```

3. **完成阶段：**
   ```bash
   git push origin feature/new-kdtree-optimization
   # 在 GitHub 上创建 PR（可选）
   ```

4. **合并回主分支：**
   ```bash
   git checkout main
   git merge feature/new-kdtree-optimization
   git push origin main
   ```

## 最佳实践

### 提交信息规范
```
feat: 新功能
fix: 错误修复
docs: 文档更新
style: 格式修改
refactor: 代码重构
test: 测试相关
chore: 构建/工具相关
```

### 文件组织
- 保持工作区整洁
- 及时提交和推送
- 使用 `.gitignore` 忽略不需要的文件

### 备份策略
- 定期推送到远程仓库
- 重要的实验分支也要推送
- 考虑使用标签标记重要版本

## 故障排除

### 常见问题和解决方案

1. **合并冲突：**
   - 手动编辑冲突文件
   - 使用 VS Code 的合并工具
   - 测试合并后的代码

2. **意外提交：**
   - 使用 `git reset` 撤销提交
   - 使用 `git reflog` 查看操作历史

3. **分支混乱：**
   - 使用 `git branch -a` 查看所有分支
   - 清理不需要的分支

记住：Git 很强大，但也很安全。大多数操作都可以撤销，所以大胆尝试！