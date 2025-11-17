# 超简单 Git 个人开发指南

> 专为 Git 初学者设计的极简工作流，只使用最基础的命令

## 你的现状

✅ **已经完成：**
- 从 GitHub 克隆了代码到本地
- 已经做了一些修改
- 远程仓库：origin → https://github.com/tanrenjun/RoGS.git (你的 fork)

## 最基础的概念

### 三个区域
```
工作区 ←→ 暂存区 ←→ 本地仓库 ←→ 远程仓库
(修改)   (准备)    (提交)      (推送)
```

### 最常用命令（只需要记住这5个）
```bash
git status     # 查看状态（随时用）
git add .      # 添加所有修改到暂存区
git commit -m "描述"  # 提交到本地仓库
git push       # 推送到 GitHub
git log        # 查看提交历史
```

## 个人开发工作流

### 🌟 最简单的日常流程

#### 1. 开始工作
```bash
git status                    # 看看有什么修改
```

#### 2. 保存工作（随时做）
```bash
git add .                     # 把所有修改加入暂存区
git commit -m "今天的工作"     # 提交到本地
git push                      # 推送到你的 GitHub
```

#### 3. 查看历史
```bash
git log                      # 看看之前的提交
```

### 📝 实际演示

让我们用你刚才的修改来练习：

#### 步骤 1: 查看当前状态
```bash
git status
```
你会看到很多红色的"修改"和"未跟踪文件"

#### 步骤 2: 选择要保存的文件（两种方法）

**方法A：保存所有文件**
```bash
git add .                    # 所有文件都保存
```

**方法B：只保存特定文件**
```bash
git add models/road.py       # 只保存这个文件
git add 文件名               # 只保存指定文件
```

#### 步骤 3: 提交（写一句描述）
```bash
git commit -m "添加了代码注释"
```

#### 步骤 4: 推送到你的 GitHub
```bash
git push
```

### 🔄 个人分支工作流（推荐）

#### 为什么用分支？
- main 分支保持稳定
- 每个功能单独开发
- 不会弄乱主代码

#### 超简单分支流程

1. **创建个人工作分支**
```bash
git checkout -b my-work      # 创建并切换到新分支
```

2. **在这个分支上工作**
```bash
# ... 修改代码 ...
git add .
git commit -m "工作进度"
git push                     # 第一次会提示设置上游
```

3. **如果提示设置上游，运行：**
```bash
git push --set-upstream origin my-work
```

4. **以后继续工作**
```bash
git checkout my-work         # 切换到工作分支
git add .
git commit -m "继续工作"
git push                     # 直接推送
```

## VS Code 图形操作（不用记命令）

### 基本操作
1. **打开 Git 面板**: Ctrl+Shift+G
2. **查看修改**: 点击文件名看具体改了什么
3. **暂存文件**: 点击文件右侧的 `+` 号
4. **提交**: 输入消息，按 Ctrl+Enter
5. **推送**: 点击"..."菜单 → "推送"

### 分支操作
1. **左下角点击分支名**
2. **选择"创建新分支"**
3. **输入分支名**（如：my-development）
4. **在新分支上工作**

## 常见问题解决

### ❓ "git push 失败"
```bash
# 第一次推送新分支时
git push --set-upstream origin 分支名
```

### ❓ "想撤销修改"
```bash
# 撤销某个文件的修改
git checkout -- 文件名

# 撤销所有修改
git checkout -- .
```

### ❓ "提交信息写错了"
```bash
# 修改最后一次提交信息
git commit --amend -m "新的描述"
```

### ❓ "想回到之前的状态"
```bash
# 查看提交历史
git log

# 回到某个提交（小心使用）
git reset --hard 提交ID
```

## 推荐的工作习惯

### 每日习惯
1. **开始工作前**: `git status` 看看状态
2. **工作中**: 经常 `git add .` + `git commit` 保存进度
3. **结束工作**: `git push` 推送到 GitHub

### 文件组织
1. **main 分支**: 保持稳定，作为备份
2. **个人分支**: 日常开发（如：my-dev）
3. **实验分支**: 尝试新功能（如：experiment-xxx）

## 超简化的命令总结

```bash
# 每天必用
git status          # 看状态
git add .           # 准备提交
git commit -m "描述" # 提交
git push            # 推送

# 分支操作
git checkout -b 新分支名    # 创建新分支
git checkout 分支名         # 切换分支
git branch                  # 查看分支

# 查看历史
git log             # 提交历史
git log --oneline   # 简洁历史
```

## 🎯 下一步建议

1. **现在就用你的修改练习**: 按照上面的步骤提交你的代码注释
2. **创建一个个人分支**: `git checkout -b my-development`
3. **以后都在这个分支工作**: 养成习惯
4. **经常 push**: 保持 GitHub 备份最新

记住：**Git 很安全，大胆尝试！** 大多数操作都可以撤销，不要害怕犯错。从简单的开始，慢慢就熟练了！

## 📞 需要帮助时

如果命令不记得：
1. 先用 `git status`，Git 会提示你该怎么做
2. 用 VS Code 的图形界面（Ctrl+Shift+G）
3. 查看这个指南
4. 随时问我！