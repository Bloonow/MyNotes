# Git

Git是一个分布式版本控制系统，具有丰富的命令集，可提供高级操作和对内部的完全访问。版本控制是基于工作目录的，每次目录状态的变化，无论是新建或删除文件，亦或是修改文件内容，都可以称为是一个版本。Git版本控制系统可以记录工作目录的所有状态变化，并支持回滚到历史上的某一个版本。

Git使用三树原理来管理版本变化，包括工作区树（working tree）、暂存区树（staging area/index）、版本库树（commit/HEAD），可使用treeish树对象代指这三种树。

工作区树（working tree），工作区树即是实际文件系统中的项目目录，是工作时实际操作和修改的目录，是项目的当前状态。

暂存区树（staging area/index），暂存区树是一个中间临时区域，在对工作区树做出修改但还没有提交修改时，可以使用git add命令将工作区的修改添加到暂存区，暂存区树保存即将提交的修改。

版本库树（commit/HEAD），版本库树是存储所有历史记录和版本的目录结构，每做出一次提交，都会在版本库中创建一个新的版本，每个版本都包含一组文件的快照，以及提交相关的元数据，如作者、时间戳和提交消息。版本库树由一系列的提交（commit）组成，最新的提交被称为HEAD头。

## 安装与LFS

在Windows平台上，可以从https://git-scm.com/downloads网址下载安装包，并安装。在安装之后，可以使用git-bash终端，也可以在CMD中使用git命令。

在Linux平台上，可以使用apt install git命令安装。在安装之后，Git套件被集成到bash shell终端中，可以直接使用git命令。

近几年，随着大模型技术的发展，越来越多的Git仓库中包含着预训练的模型文件，这些文件通常能达到GiB级别。为更好的处理Git仓库中的大文件，一个开源的扩展Git Large File Storage（LFS）对Git仓库中的大文件管理和存储进行了优化，它通过替换Git仓库中的大文件为指针文件，并将实际的文件存储在远程LFS服务器上，从而避免Git仓库体积过大，提高克隆和拉取的速度。

在Windows平台上，可以从https://git-lfs.com网站下载安装包，并安装。在Linux平台上，可以使用apt install git-lfs命令安装Git LFS扩展。

在使用Git LFS之前，需要为每个Git账户启用GIT LFS功能，执行如下命令即可，每个Git账户仅需执行一次，这会在.gitconfig文件中配置Git LFS功能。

```shell
git lfs install
```

在仓库项目中，使用如下命令添加要以LFS模式跟踪的大文件，也可以直接编辑.gitattributes配置文件。

```shell
git lfs track "*.psd"
```

在使用LFS跟踪之后，会在仓库项目的目录中创建.gitattributes配置文件（如果不存在），并在文件中添加如下内容。

```
*.psd filter=lfs diff=lfs merge=lfs -text
```

之后需要Git仓库跟踪.gitattributes配置文件，如下所示。

```shell
git add .gitattributes
```

然后就可以正常使用Git命令，而对于诸如之前配置的*.psd文件，Git LFS会自动管理。在使用git push命令时，Git LFS会将所管理的大文件上传到LFS服务器。在使用git clone命令或git pull命令时，Git LFS会自动下载相应的大文件。当然，也可以使用git lfs xxx系列命令手动管理大文件。

## 设置与配置

### git config

```shell
git config
```

使用git config命令可以查询/设置/替换/撤销某个配置，配置文件分为系统级别（/etc/gitconfig）、用户全局级别（\$HOME/.gitconfig或\$HOME/.config/git/config）、本地仓库级别（\$REPO/.git/config）三个层次，分别对应--system、--global、--local三个选项。

在读取配置文件时默认以系统、全局、本地仓库的顺序读取，在写入配置文件时默认写入本地仓库级别的配置文件。

在使用Git时首先要设置用户名和电子邮箱，以标识进行操作的用户，如下所示。

```shell
git config --global user.name "Bloonow@PC"
git config --global user.email "bloonow@163.com"
```

可使用--list选项查看所设置的配置文件。

```shell
git config --list
```

## 创建与克隆

### git init

```shell
git init
```

该命令在工作目录中创建一个空的Git存储库（repository），本质上是创建一个.git目录，其中包括config、description、index、HEAD文件，branches、hooks、logs、objects、refs目录等。

使用--initial-branch选项可指定所创建仓库的初始分支名称，若未指定则默认采用master名称。在已存在Git仓库的工作目录下使用git init命令是安全的，不会覆盖已存在的文件。

config文件是Git仓库的配置文件，用于存储仓库级别的配置选项。该文件是一个文本文件，使用INI格式（键值对）来组织配置信息。

description文件是一个纯文本文件，用于提供关于Git仓库的简要描述信息。

index文件是一个二进制文件，也称为暂存区（staging area）或者索引（index），用于记录当前工作目录中所修改文件的快照信息。当使用git add命令将所修改的文件添加到暂存区时，Git会将这些文件的快照信息保存在index文件中。index文件的内容包括文件名称、文件元数据（如权限和时间戳）、文件内容的哈希值。它充当工作目录和下一次提交之间的桥梁。

HEAD文件是一个特殊的引用文件，用于指示当前所在分支或提交。该文件的内容可以是refs/heads/branch_name的形式，表示当前所在的分支；或者是直接指向某个提交的哈希值，表示当前处于分离头指针（detached HEAD）状态，即不在任何分支上工作。通过查看HEAD文件的内容，可以确定当前所在的分支或直接指向的提交，这对于了解当前工作状态、进行分支操作和切换等非常有用。

branches目录是一种不常用的存储速记方式，用于指定git fetch、git pull、git push的URL，目前已基本不用。

hooks目录用于存储Git的一些可执行脚本，它们在特定的Git操作（如提交、合并、推送等）之前或之后执行，可用于自定义和控制Git的行为。

logs目录用于存储Git仓库的引用（HEAD、分支、标签等）的历史日志信息，即记录refs目录的历史状态。这些日志文件记录引用的变动历史，包括引用的新增、删除、移动等操作，它们可以用于查看仓库中引用的修改历史，以及恢复到之前的引用状态。

objects目录用于存储Git仓库中的所有对象，包括提交对象、树对象、文件对象等。Git默认使用SHA-1哈希算法获得每个对象的唯一标识（40位十六进制数），然后将前2位作为目录名称，将后38位作为文件名称，将对象存储到对应目录下的文件中。每个对象都以二进制格式存储，文件内容经过压缩和哈希计算。

> SHA-1（Secure Hash Algorithm 1）安全散列哈希算法是一种密码散列函数，可生成一个被称为消息摘要的160位二进制散列值，散列值通常的呈现形式为40个十六进制数。

objects目录还包含其它目录。info子目录存储一些辅助信息和索引文件，用于加快对象访问速度。pack子目录存储定期打包的对象文件，Git的打包机制会定期将一些对象文件打包成一个单独文件，并使用压缩算法来减小存储空间和提高性能。

refs目录用于存储指向提交对象的引用（reference）。heads子目录存储分支引用，每个分支都对应一个文件，文件名与分支名称相同，这些文件中的内容是指向分支最新提交的指针。tags子目录存储标签引用，每个标签都对应一个文件，文件名与标签名称相同，这些文件中的内容是指向标签的对象的指针。remotes子目录存储远程引用，每个远程仓库都对应一个子目录，目录名与远程仓库名称相同，在每个远程仓库目录下，可以存储与该远程仓库相关的引用，如远程分支引用。这些引用文件（通常是文本文件）记录指向特定提交对象的指针，通过这些引用，Git可以跟踪和管理分支、标签以及与远程仓库的交互。

### git clone

```shell
git clone repository [path]
```

该命令将一个远程仓库repository克隆到path目录中。根据克隆仓库中的分支，在新仓库中创建它们的远程跟踪分支（remote-tracking）；根据克隆仓库的当前活动分支，为新仓库创建一个初始分支，即创建新的HEAD指向克隆仓库的HEAD所指向的分支。

在创建之后，使用无参数的git fetch命令更新所有远程跟踪分支，使用无参数的git pull命令会将可能存在的远程主分支合并到当前主分支。

## 忽略跟踪

在团队协作的项目中，有一些诸如本地配置文件是所有团队成员都拥有的，但此类文件无需跟踪并推送，可使用.gitignore文件指定匹配模式，以忽略对某些路径对象（目录或文件）的跟踪，已被Git追踪的路径不受影响，详见下面的解释。

在.gitignore文件中，每一行条目指定一个匹配模式，以指定需要忽略的路径对象。Git通常会检查多个来源的.gitignore模式，优先级从高到低如下所示，同一优先级则由最后的匹配模式决定。

1. 从支持忽略规则的命令行中读取模式。
2. 从当前路径目录下的.gitignore文件中，或从任何父目录（直到工作区树顶层目录）中读取模式，父目录.gitignore文件中的模式会被子目录.gitignore文件中的模式覆盖，直到包含目标路径对象的目录。这些模式相对于其.gitignore文件的路径位置，以相对路径进行匹配。项目通常会在其资源库中包含此类.gitignore文件，其中包含项目构建过程中生成的文件的模式。
3. 从.git/info/exclude文件中读取模式。
4. 从配置变量core.excludesFile指定的文件中读取模式。

将忽略规则的匹配模式放入哪个忽略文件取决于模式的使用方式，以下是一些常见情况。

- 需要受Git版本控制并通过克隆分发到其他仓库的路径对象，但所有团队成员都需要忽略的文件（例如本地配置文件），则应该将忽略模式放入.gitignore文件。
- 特定于某个存储仓库并且无需与其他仓库共享的路径对象（例如存在于仓库内部但特定于某个用户工作流程的辅助文件），则应该将忽略模式放入.git/info/exclude文件。
- 用户希望Git在任何情况下都进行忽略的匹配模式（例如由用户的编辑器软件生成的备份或临时文件），一般会将忽略模式放入到由用户家目录\$HOME/.gitconfig中的core.excludesFile条目所指定的文件中，默认是\$XDG_CONFIG_HOME/git/ignore文件。如果XDG_CONFIG_HOME环境变量未设置或为空，则使用\$HOME/.config/git/ignore文件代替。

值得注意的是，底层的Git工具，如git ls-files命令和git read-tree命令，会读取命令行选项指定的.gitignore模式，或从命令行选项指定的文件中读取。而更高层次的Git工具，如git status命令和git add命令，会使用上述指定来源的模式。

下面介绍.gitignore所使用的用于忽略跟踪的匹配模式串的格式，空行不匹配任何文件，因此可以作为分隔符以提高可读性，使用井号#进行注释。

- 模式串尾部的空格将被忽略，除非使用反斜杠`\`进行转义。
- 使用斜杆`/`作为目录分隔符，可以出现在模式串的开头、中间、结尾。如果分隔符/出现在模式串的开头或中间，则匹配基于.gitignore文件的路径位置开始，并以相对路径进行匹配。如果分隔符/出现在模式串的结尾，则匹配一个目录以及目录下的对象。例如，doc/会匹配doc目录以及release/doc等目录；而doc/main/则只会匹配doc/main目录而不会匹配release/doc/main目录，因为分隔符出现在模式串的中间，匹配是基于.gitignore的路径位置开始的。
- 星号`*`匹配斜线/以外的任意字符，字符`?`匹配斜线/以外的一个任意字符，符号范围诸如[a-zA-Z]匹配范围内的一个任意字符。
- 双星号`**`在路径匹配中存在特殊用法。在分隔符之前的诸如`**/`表示在所有目录中匹配，在分隔符之后的诸如`/**`表示匹配目录中的所有对象。例如，\*\*/doc匹配任何目录中的doc对象，doc/\*\*匹配基于.gitignore路径位置的doc目录中的所有对象。
- 一个可选的前缀`!`，用于否定模式，任何被先前模式排除的匹配文件都将被重新包含。如果该文件的父目录已被排除，则无法重新包含该文件。如果一个匹配规则的模式串以!开头，在需要第一个!前面加反斜杠，例如\\!important!.txt模式串。

需要注意，使用.gitignore文件的目的是确保某些不被Git跟踪的路径对象（目录或文件）不被跟踪，要停止跟踪当前已被跟踪的文件，可使用git rm --cached path命令从index索引中移除该文件。访问工作区树中的.gitignore文件时，Git不会跟踪符号链接。

## 基本快照

### git add

```shell
git add [path]
```

该命令在工作区树（working tree）中找到path所匹配的当前内容（current content）用来更新index索引，为下一次提交准备暂存的内容。通常情况下，该命令会将已存在的路径作为一个整体进行添加，可使用选项仅添加对工作区树文件所做的部分更改，或删除工作区树中不存在的路径。

> 当前内容（current content）是指相对于上一次提交，目录状态的所有变化，包括新建、删除、修改的所有对象。

索引保存着工作区树内容的快照，该快照作为下一次提交的内容。因此，在对工作区树进行任何更改之后，必须使用git add命令将所有新文件或修改过的文件添加到索引中，才能使用git commit命令进行提交。

使用path参数指定要添加的文件路径或目录路径，指定目录时会递归添加所有子文件和子目录，指定路径为`.`时会添加当前工作目录下所有修改对象，或使用-A选项添加所有修改。

### git status

```shell
git status [path]
```

该命令显示index索引和HEAD当前提交之间存在差异的对象路径，需要使用git commit命令提交；显示工作区树和index索引之间存在差异的对象路径，以及工作区树中没有被Git跟踪的路径，需要使用git add命令暂存并使用git commit命令提交。

使用--short选项以简短格式显示输出。

```shell
vim a.c
vim b.cpp
git add .
gcc a.c -o a.exe
g++ b.cpp -o b.exe
git status
```

```shell
On branch master

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
	new file:   a.c
	new file:   b.cpp

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	a.exe
	b.exe
```

### git commit

```shell
git commit [-m message]
```

该命令创建一个提交，包括index索引的当前内容，和log信息所描述的变化。新的提交是HEAD的直接子节点，通常是当前分支的顶端，然后HEAD头会被更新并指向新的顶端。注意，未被任何分支指向的提交是处于游离状态的，会被Git垃圾回收机制删除。

使用-m选项指定对该次提交进行描述的信息，必须指定该选项。使用--amend -m选项可以修改最后一次提交的注释。

使用-a选项，可以使Git自动添加所有已知文件的修改，即所有已经列在索引中的文件，并自动删除索引中已经从工作区树中删除的文件，然后执行实际的提交。

```shell
git commit -a -m "first commit"
git status
```

```shell
On branch master
Untracked files:
  (use "git add <file>..." to include in what will be committed)
	a.exe
	b.exe

nothing added to commit but untracked files present (use "git add" to track)
```

### git reset

```shell
git reset [treeish] [path]
```

该命令将path匹配的所有index索引文件设置为树treeish时的状态，并不会影响工作区树和HEAD当前分支，它是git add命令的逆操作。

其中，树treeish可以是一个commit提交对象，或一个branch分支对象，默认是HEAD头所指向的当前分支。

```shell
git status
```

```shell
On branch master
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
	new file:   c.py
```

```shell
git reset .
git status
```

```shell
On branch master
Untracked files:
  (use "git add <file>..." to include in what will be committed)
	c.py

nothing added to commit but untracked files present (use "git add" to track)
```

此外，命令git reset还能用一个commit提交对象作为参数。

```shell
git reset [option] [commit]
```

该命令将当前分支头HEAD重载为提交commit时的状态，并根据option选项决定更新index索引文件和工作区树的状态。使用--soft选项，完全不影响index文件和工作区树；使用--mixed选项，会重载index文件但不会影响工作区树，这是默认选项；使用--hard选项，会重载index文件和工作区树；使用--merge选项，会重载index索引并更新工作区树，用于撤销一个合并然后恢复到commit状态。

### git rm

```shell
git rm [path]
```

该命令可以从index索引文件和工作区树中删除path匹配的所有对象，使用-r选项递归地删除一个目录及其所有子文件和子目录，使用--cache选项可以只从index索引中删除，而不影响工作区树。

### git mv

```shell
git mv [src_path] [dst_path]
```

该命令可以将src_path匹配的所有对象移动到dst_path路径下，可进行重命名操作。在移动完成后，index索引就会被更新，但更改仍需要进行提交才能生效。

## 查看与比较

### git log

```shell
git log [commit]
```

该命令显示直到commit之前的提交记录，即所有能够到达commit的父提交记录，默认是显示所有能够到达HEAD的父提交。参数commit可以指定为某个提交对象的哈希值，显示所有能够达到指定commit对象的所有提交记录。此外，可使用HEAD表示当前分支或提交。

> 对于某个commit提交对象，可使用commit\^表示它的父提交，即它的直接上一次提交，若存在多个父提交，可使用commit\^#的形式指定（其中#是从1开始的数字）；也可使用commit\~#的形式，其中\~#等价于指定#个\^符号，#缺省时表示数字1。

使用--oneline选项简化输出，使用--graph命令打印仓库当前分支提交的树状图。

```shell
git branch --list | xargs -n 1 | grep [^*] | xargs git log --oneline --graph
```

上述命令可以查看当前仓库的所有提交。

### git diff

```shell
git diff [arg] [arg]
```

该命令显示两个arg之间的状态变化，其中path可以指定为index索引、工作区树、指定目录、指定文件、指定合并等。

### git blame

```shell
git blame filepath
```

该命令用于显示filepath所指定的文本文件的内容，并在每行内容之前显示最后一次修改该行时的注释信息。默认会显示完整的文件内容，使用-L #,#选项指定所要显示的行的起始行号；使用-n选项显示文件内容的行号。

```shell
git blame -n main.cpp
```

```
5bd30ad6 1 (Bloonow 2024-04-11 00:32:24 +0800 1) #include <cstdio>
5bd30ad6 2 (Bloonow 2024-04-11 00:32:24 +0800 2) #include <cstdlib>
445e3e28 3 (Bloonow 2024-04-11 00:35:18 +0800 3) #include <iostream>
445e3e28 4 (Bloonow 2024-04-11 00:35:18 +0800 4) using namespace std;
5bd30ad6 3 (Bloonow 2024-04-11 00:32:24 +0800 5) int main() {
445e3e28 6 (Bloonow 2024-04-11 00:35:18 +0800 6)     // printf("Hello World!\n");
445e3e28 7 (Bloonow 2024-04-11 00:35:18 +0800 7)     cout << "Hello World!" << endl;
5bd30ad6 5 (Bloonow 2024-04-11 00:32:24 +0800 8)     return 0;
5bd30ad6 6 (Bloonow 2024-04-11 00:32:24 +0800 9) }
```

## 分支与合并

分支（branch）依附于一个提交对象（commit）上，它只是指向某个提交对象的指针，删除这个branch指针并不会影响commit提交对象本身。

标签（tag）依附于一个提交对象（commit）上，它只是指向某个提交对象的指针，删除这个tag指针并不会影响commit提交对象本身。

分支branch与标签tag的区别在于，标签tag常用在项目开发的重要版本上，例如软件版号v1.0.0之类的标签。

### git branch

```shell
git branch [--list] [pattern]
```

该命令的无参数版本或使用--list选项会列出现有分支，当前分支以绿色突出显示并标有`*`星号，而其他任何分支以青色显示并标有`+`加号。使用-r选项显示远程跟踪分支；使用-a选项显示本地分支和远程分支；使用-v选项显示分支名称、哈希散列值；使用-vv选项显示分支名称、哈希散列值、所跟踪的上游远程分支；使用pattern参数匹配相应的分支。

```shell
git branch branch_name [start_point]
```

该命令创建一个名称为branch_name的新分支，并且指向start_point起始点，它可以是一个分支名称、提交对象的哈希散列值、标签。若不指定start_point则新的分支会指向HEAD当前分支。

```shell
git branch --track branch_name [start_point]
```

该命令创建一个名称为branch_name的新分支，并使用--track选项指定一个start_point起始点（通常是远程分支），作为新创建分支branch_name的上游跟踪分支。

```shell
git branch --set-upstream-to=<upstream> branch_name
```

该命令为已存在的分支branch_name设置其上游远程跟踪分支upstream，例如将上游远程跟踪分支设置为origin/master分支。如果未指定branch_name参数则为HEAD头所指向的当前分支设置上游远程跟踪分支。

在为一个本地分支branch_name设置上游远程跟踪分支remote_name/remote_branch时，会将本地分支的branch.branch_name.merge配置赋值为refs/heads/remote_branch，表示上游远程跟踪分支，将将本地分支的branch.branch_name.remote配置赋值为remote_name，表示远程存储仓库名称。

```shell
git branch [-m|M|c|C] [old_branch] new_branch
```

使用-m或-M选项将旧分支old_branch移动（重命名）为新分支new_branch；使用-c或-C选项将旧分支复制为新分支new_branch。当未指定old_branch分支时，默认使用HEAD头所指向的当前分支。

```shell
git branch [-d|D] [-r] branch_name
```

使用-d或-D选项删除一个分支branch_name，同时使用-r选项删除远程跟踪的分支，只有当远程分支不再存在于远程仓库或设置git fetch不再获取它们时，删除远程分支才有意义。

### git checkout

```shell
git checkout branch_name
```

该命令更改当前工作区树，并使当前分支HEAD头重新指向branch_name作为当前分支（即切换分支），并将工作区树切换到对应分支branch_name的状态。

```shell
git checkout [treeish] [path]
```

该命令更改path所匹配的当前工作区树文件，并使当前分支HEAD头重新指向，未指定path时匹配整个当前工作区树。当指定树treeish参数时，将index索引文件和当前工作区树文件更新为treeish指定状态。当未指定树treeish参数，使用index索引文件更新当前工作区树。

其中树treeish可以是index索引文件，或一个commit提交对象，或一个branch分支对象，默认是HEAD头所指向的当前分支。

若treeish是一个之前的提交对象，则会使HEAD头处于游离状态（即不指向任何分支）。

```shell
git log --oneline HEAD master
```

```shell
04f20b5 (HEAD -> master, b3) commit3
ddfb2bd commit2
e63169c commit1
```

```shell
git checkout HEAD^^
git log --oneline HEAD master
```

```shell
04f20b5 (master, b3) commit3
ddfb2bd commit2
e63169c (HEAD) commit1
```

```shell
vim f1_1.c
git add .
git commit -m "commit1_1"
git branch b1
git checkout master
git log --oneline --graph HEAD master b1
```

```shell
* b63313c (b1) commit1_1
| * 04f20b5 (HEAD -> master, b3) commit3
| * ddfb2bd commit2
|/  
* e63169c commit1
```

```shell
git checkout [--ours|--theirs] path
```

该命令用于在出现合并冲突时，选择保留当前分支（--ours）或并入分支（--theirs）的path文件对象。

### git switch

```shell
git switch [brach_name]
```

该命令将HEAD头切换到指定分支brach_name，工作区树和index索引会同步更新为与新分支一样的状态，所有新的提交都会被添加到该分支顶端。

> 这个命令是实验性的，其行为可能会发生改变。

### git merge

```shell
git merge [-m message] [commit]
```

该命令将所指定的commit提交中的变化内容（从指定提交在历史上与当前分支分离时起）合并入当前分支HEAD，并创建一个新的提交，使用-m选项指定新提交的信息。

其中，提交commit通常是一个需要被合并的其它分支，如果未指定commit提交或分支，则合并当前HEAD分支的上游远程跟踪分支。

当被合并的提交或分支是当前分支HEAD的后代，即被合并分支比当前分支新时，默认使用--ff选项指定的快进（fast-forward）方式进行合并。即只更新当前分支HEAD指针（与index索引文件）以匹配合并后的分支，即修改当前分支HEAD指针到更新的被合并分支上，不创建提交。当被合并分支不是当前分支后代时，则会创建一个新的合并提交。

```shell
git log --oneline --graph HEAD master b1
```

```shell
* b63313c (b1) commit1_1
| * 04f20b5 (HEAD -> master, b3) commit3
| * ddfb2bd commit2
|/  
* e63169c commit1
```

```shell
git merge -m "merge1" b1
git log --oneline --graph HEAD master b1
```

```shell
*   c21d40b (HEAD -> master) merge1
|\  
| * b63313c (b1) commit1_1
* | 04f20b5 (b3) commit3
* | ddfb2bd commit2
|/  
* e63169c commit1
```

当出现合并冲突时，可手动选择所保留的修改，或使用git mergetool工具来解决合并冲突。

### git stash

```shell
git stash command
```

该命令可以记录index索引和工作区树的当前状态（作为贮藏区），并返回到HEAD头所指向的当前分支或当前提交，会影响index索引和工作区树。

```shell
git stash list
```

```shell
git stash show
```

使用git stash list命令列出所有贮藏的内容。使用git stash show命令显示贮藏内容所进行的修改。

```shell
git stash push [path]
```

```shell
git stash pop
```

使用git stash push将path所匹配的所有文件对象提交到贮藏区，并重载index索引和工作区树到HEAD头所指向的状态。使用git stash pop命令将贮藏区状态恢复到当前index索引和工作区树上。

```shell
git stash clear
```

使用git stash clear清除贮藏区。

```shell
git status --short
```

```shell
A  tmp.py
```

```shell
git stash push .
```

```shell
Saved working directory and index state WIP on master: c21d40b merge1
```

```shell
git stash list
git stash show
```

```shell
stash@{0}: WIP on master: c21d40b merge1
 tmp.py | 1 +
 1 file changed, 1 insertion(+)
```

### git tag

```shell
git tag [-m message] tag_name [commit]
```

该命令创建一个指向commit提交的标签，默认commit为HEAD头指向当前分支或当前提交。

使用-m选项指定该标签的描述信息，这种创建的标签为轻量级标签，此时隐含指定-a选项创建无签名有注释的标签对象。

使用-s选项创建GPG签名标签；使用-u选项创建使用给定密钥的GPG签名标签；这种方式创建的标签包含创建日期、标记者名称、电子邮件、GPG签名等，用于发布。

使用--list选项列出标签；使用-v选项验证指定的标签；使用-d选项删除一个标签。

```shell
git tag -m "first tag" v0.1.0
git log --oneline --graph HEAD
```

```shell
*   c21d40b (HEAD -> master, tag: v0.1.0) merge1
|\  
| * b63313c (b1) commit1_1
* | 04f20b5 (b3) commit3
* | ddfb2bd commit2
|/  
* e63169c commit1
```

### 合并冲突与解决

假设存在一个初始仓库，拥有一个默认主分支master，且仅有一个hello.py文件，内容如下所示。

```python
name = input('Enter your name: ')
print(f'Hello, {name}!')
```

现创建一个新分支devp并切换到该分支，向工作区树目录中添加一个res.tgz资源文件，并修改hello.py文件内容，如下所示。

```python
import math
name = input('Enter your last name: ')
print(f'Hello, {name}!')
# New Features Below
number = round(math.pi * len(name))
print(f'Your luck number is {number}')
```

提交devp分支，切换到master分支，同样添加一个名为res.tgz但内容不同的资源文件，并修改hello.py文件内容，如下所示。

```python
name = input('Enter your name: ')
# New Features Below
print(f'Do you love your name {name}?')
print(f'The length of your name is {len(name)}')
```

显示当前存储仓库的分支日志，如下所示。

```shell
git log --oneline --graph master devp
```

```shell
* 4ede85a (HEAD -> master) master: update
| * 397a7cb (devp) devp: update
|/  
* f607ce5 initialization
```

现在考虑将devp分支合并到master分支中并创建一个新的合并提交，可以预见的，肯定会出现合并冲突。

值得注意的是，在合并外部分支或提交时，最好先将自己的分支进行一次提交，这样即使合并失败，也无需担心所做的工作被破坏；也可使用git stash命令将当前index索引和工作区树保存到贮藏区。

在进行合并操作时，Git会调和（reconcile）所有分支的更改，并更新HEAD头、index索引文件、工作区树目录状态。只要工作区树目录的更改不重叠，Git就可以保留所有更改。而当如何调和并不明显时，会发生如下情况，其中MERGE_HEAD引用和AUTO_MERGE引用与HEAD头一样，是位于.git目录中的文件。

1. HEAD头指针保持不变，MERGE_HEAD引用指向要被合并的分支头。
2. 无重叠的干净路径会在index索引文件和工作区树中更新。
3. 对于冲突路径，index索引文件会记录三个版本。阶段1记录共同祖先的版本，阶段2记录HEAD的版本，阶段3记录MERGE_HEAD的版本。可以使用git ls-files -u命令查看这些阶段。当前工作区树目录会包含合并操作的结果，并使用冲突标记`<<<`、`===`、`>>>`标识三个阶段的冲突。
4. 合并结果会写入一个AUTO_MERGE引用，指向与工作区树当前内容相对应的树，其中包含文本冲突的冲突标记。仅当使用ort合并策略时才会写入此AUTO_MERGE引用（默认）。
5. 除此之外，并无其他更改。特别是，在执行合并操作之前的局部修改，与它们的index条目将保持不变，也即匹配HEAD头所指向的状态。

当多个分支由于重叠更改导致合并冲突时，Git将使用冲突标记呈现冲突。针对上述示例进行合并，结果如下所示。

```shell
git merge -m "merge devp to master" devp
```

```shell
warning: Cannot merge binary files: res.tgz (HEAD vs. devp)
Auto-merging hello.py
CONFLICT (content): Merge conflict in hello.py
Auto-merging res.tgz
CONFLICT (add/add): Merge conflict in res.tgz
Automatic merge failed; fix conflicts and then commit the result.
```

```shell
git status
```

```shell
On branch master
You have unmerged paths.
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)

Unmerged paths:
  (use "git add <file>..." to mark resolution)
        both modified:   hello.py
        both added:      res.tgz

no changes added to commit (use "git add" and/or "git commit -a")
```

```shell
cat hello.py
```

```python
import math
name = input('Enter your name: ')
<<<<<<< HEAD
# New Features Below
print(f'Do you love your name {name}?')
print(f'The length of your name is {len(name)}')
=======
print(f'Hello, {name}!')
# New Features Below
number = round(math.pi * len(name))
print(f'Your luck number is {number}')
>>>>>>> devp
```

当合并操作出现冲突时。如果不想再进行合并，可使用git merge --abort命令恢复到合并之前的状态。如果想继续合并，则需要在出现冲突的位置选择所要保留的修改，手动解决合并冲突，也可使用git mergetool启动图形界面的合并工具，协助解决合并冲突。如果想直接保留某一方修改的整个文件而不是逐一选择某行采用哪一方的修改，或者是遇到无法逐行查看修改的非文本二进制文件时，可使用git checkout --ours path保留本分支的文件对象，或使用git checkout --theirs path保留并入分支的文件对象。

合并冲突解决后，使用git add命令将修改添加到index索引文件条目中，然后可使用git merge --continue命令或直接使用git commit命令提交所做的修改。其中git merge --continue命令会先检查当前是否正在进行一个合并，然后再调用git commit命令。

如上示例，解决冲突（选择保留当前修改）之后，合并完成之后状态如下。

```shell
git log --oneline --graph master devp
```

```shell
*   9f5b1a7 (HEAD -> master) merge devp to master
|\  
| * 397a7cb (devp) devp: update
* | 4ede85a master: update
|/  
* f607ce5 initialization
```

## 共享与更新

### git remote

```shell
git remote command
```

该命令用于管理（远程）仓库的集合，用于跟踪这些仓库的分支。该命令不带参数或使用-v选项，用于列出现有的远程仓库列表。

```shell
git remote add remote_name URL
```

使用git remote add命令将URL远程存储仓库添加为remote_name名称，通常采用origin作为远程仓库的名称。然后可使用git fetch命令创建refs/remotes/remote_name/branch_name远程跟踪分支并更新。

```shell
git remote remove remote_name
```

```shell
git remote rename old_name new_name
```

使用git remote remove命令可以删除名称为remote_name的远程仓库，该远程仓库的所有远程跟踪分支和配置设置都将被删除。使用git remote rename命令将重命名远程仓库，该远程仓库的所有远程跟踪分支和配置设置都将被更新。

### git fetch

```shell
git fetch [repository] [refspec]
```

该命令从远程存储仓库repository中获取（fetch）分支和标签，以及构建这些分支和标签完整历史所需的对象。默认情况下，任何指向被获取历史的标签也会被获取，使用--tags选项从远程获取所有标签，将远程标签获取为同名的本地标签，即refs/tags/\*:refs/tags/\*。

从远程仓库中获取的分支和标签的名称，会存储到本地FETCH_HEAD文件中，并且会创建refs/remotes/remote_name/branch_name远程跟踪分支，并更新这些远程跟踪分支。远程存储仓库的HEAD头所指向的远程仓库的当前分支信息，会存储到本地ORIG_HEAD文件中。

远程存储仓库repository可以是一个在git remote命令中配置的仓库名称，或直接是一个远程仓库的URL地址，使用--multiple选项可从多个仓库中获取。当未指定repository参数时，默认使用origin远程存储仓库。

分支和标签统称为refs引用对象，由命令中的refspec参数指定，形式如下。

```shell
[+][src[:dst]]
```

其中，加号+表示需要进行非快进式操作，源src表示要获取的远程仓库中的引用，目标dst表示要更新的本地仓库中的引用（不存在则会创建）。它们都可以是一个对象的完整哈希散列值，也可以是分支或标签的名称，并且可以使用通配符。例如，refs/heads/\*:refs/remotes/origin/\*，表示获取远程仓库中的refs/heads/\*引用，然后更新到本地仓库的refs/remotes/origin/\*引用当中。

当只指定参数的src项而省略dst项时，则使用诸如refs/heads/src:refs/remotes/repository/src的形式进行匹配并更新。当省略refspec参数时，使用remote.repository.fetch配置，其中内容如下所示。

```shell
git config --list
```

```shell
remote.origin.fetch=+refs/heads/*:refs/remotes/origin/*
```

```shell
git remote add origin git@github.com:scaomath/galerkin-transformer.git
git fetch
```

```shell
remote: Enumerating objects: 650, done.
remote: Counting objects: 100% (23/23), done.
remote: Compressing objects: 100% (20/20), done.
remote: Total 650 (delta 7), reused 11 (delta 3), pack-reused 627
Receiving objects: 100% (650/650), 8.13 MiB | 2.95 MiB/s, done.
Resolving deltas: 100% (465/465), done.
From github.com:scaomath/galerkin-transformer
 * [new branch]      debug      -> origin/debug
 * [new branch]      main       -> origin/main
 * [new tag]         v0.1-beta  -> v0.1-beta
 * [new tag]         v0.7-beta  -> v0.7-beta
```

### git pull

```shell
git pull [repository] [refspec]
```

该命令从远程存储仓库repository中拉取（pull）分支和标签的修改，以合并入HEAD头指向的当前分支。如果当前分支落后于远程跟踪分支，默认情况下以快进（fast-forward）方式合并当前分支，以匹配远程跟踪分支。

该命令等价于先执行git fetch repository命令，再执行git merge repository/refspec命令。会将远程分支和标签存储到本地FETCH_HEAD文件，会将远程HEAD头存储到本地ORIG_HEAD文件。

远程存储仓库repository可以是一个在git remote命令中配置的仓库名称，或直接是一个远程仓库的URL地址。当未指定repository参数时，默认使用origin远程存储仓库。

分支和标签统称为refs引用对象，由命令中的refspec参数指定，形式如下。

```shell
[+][src[:dst]]
```

其中，加号+表示需要进行非快进式操作，源src表示要拉取的远程仓库中的引用，目标dst表示要更新的本地仓库中的引用（不存在则会创建）。它们都可以是一个对象的完整哈希散列值，也可以是分支或标签的名称，并且可以使用通配符。例如，refs/heads/\*:refs/remotes/origin/\*，表示获取远程仓库中的refs/heads/\*引用，然后更新到本地仓库的refs/remotes/origin/\*引用当中。

当只指定参数的src项而省略dst项时，则使用诸如refs/heads/src:refs/remotes/repository/src的形式进行匹配并更新。当省略refspec参数时，使用remote.repository.fetch配置。

需要注意的是，git pull命令在执行git fetch命令从src获取远程引用并更新到dst本地引用之后，还会执行git merge repository/src命令，以将所获取的分支合并入当前HEAD分支。而如果未指定refspec参数的src项，则会采用remote.repository.fetch配置，该配置中通常会存在通配符，则会匹配到多个远程分支，此时会终止并提示选择所要合并的远程分支。除非本地HEAD头分支已经指定上游的远程跟踪分支，则会合并上游远程跟踪分支。

```shell
git branch --track master origin/main            # this way
git branch --set-upstream-to=origin/main master  # or this
```

```shell
Branch 'master' set up to track remote branch 'main' from 'origin'.
```

```shell
git pull origin main
git checkout HEAD
```

```shell
From github.com:scaomath/galerkin-transformer
 * branch            main       -> FETCH_HEAD
 Already up to date.
 Your branch is up to date with 'origin/main'.
```

```shell
git log --oneline --graph HEAD | head -n 8
```

```shell
* f9e7d6a (HEAD -> master, origin/main) update setup.py
* 598b06b update setup.py
*   4d713e4 Merge pull request #8 from scaomath/debug
|\  
| * de63475 (origin/debug) update setup.py
| * 7af43b0 Update relative import path for pip
|/  
* f9efc8c Update README.md
```

如果尝试拉取，导致了复杂的冲突，可使用git reset命令恢复以重新开始。

如下一个示例，假设本地仓库中分支为master，并存在一个远程跟踪分支origin/master，现在远程仓库中的master分支已更新，需要将远程仓库中的分支合并到本地仓库的master分支上。

```
      A---B---C master on origin
     /
D---E---F---G master
    ^
    origin/master in local repository
```

使用git pull会获取并重放（replay）远程仓库master分支从其分离出去的位置(E)到其当前位置(C)之间的历史，然后进行合并，将结果记录到一个新的提交当中。

```
      A---B---C origin/master
     /         \
D---E---F---G---H master
```

### git push

```shell
git push [repository] [refspec]
```

该命令使用本地仓库的本地引用，来更新远程存储仓库repository中的远程引用，并同时发送完成指定引用所需的必要对象。

远程存储仓库repository可以是一个在git remote命令中配置的仓库名称，或直接是一个远程仓库的URL地址。当未指定repository参数时，则使用HEAD头所指向的当前分支的branch.branch_name.remote配置作为远程存储仓库，若未配置则默认使用origin远程存储仓库。

分支和标签统称为refs引用对象，由命令中的refspec参数指定，形式如下。

```shell
[+][src[:dst]]
```

其中，加号+表示需要进行非快进式操作，源src表示要推送的本地仓库中的引用，目标dst表示要更新的远程仓库中的引用（不存在则会创建）。它们都可以是一个对象的完整哈希散列值，也可以是分支或标签的名称，并且可以使用通配符。例如，refs/heads/\*:refs/heads/\*，表示将本地仓库中的refs/heads/\*引用，然后推送并更新到远程仓库的refs/heads/\*引用当中。

当只指定参数的src项而省略dst项时，则使用诸如refs/heads/src:refs/heads/src的形式进行匹配并更新。当省略refspec参数时，使用remote.repository.push属性的配置，但是该配置通常为空。

当refspec参数和remote.repository.push配置都未指定时，会使用HEAD头所指向的当前分支的branch.branch_name.merge配置作为远程仓库的待更新引用，通常是与本地当前分支同名的refs/heads/branch_name分支，它即是本地分支的远程上游跟踪分支。需要注意，作为一项安全措施，当上游分支与本地分支名称不一致时，推送将被终止。

```shell
echo "Hello World!" > README.md
git init
git add README.md
git commit -m "init commit"
git remote add origin git@github.com:Bloonow/MyLearn.git
git push -u origin master
```

```shell
Enumerating objects: 3, done.
Counting objects: 100% (3/3), done.
Writing objects: 100% (3/3), 219 bytes | 219.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
To github.com:Bloonow/MyLearn.git
 * [new branch]      master -> master
Branch 'master' set up to track remote branch 'master' from 'origin'.
```

使用-u选项或--set-upstream选项，对于每一个推送并更新成功的本地分支，为其设置相对应的上游远程跟踪分支。使用-d或--delete选项删除远程仓库中的某个引用（分支或标签）。

### 团队远程协作

假设远程存储仓库中存在一个developing分支，团队中的多个成员都在本地拉取该分支并进行开发，然后将新的内容推送到developing分支。对于第一次推送的人来说，没有任何问题。而对于之后推送的人来说，会发现在推送的时刻，远程仓库中developing分支的版本更新，而本地仓库中developing分支的版本更旧，此时不能使用--force选项强行覆盖掉之前成员的修改。而应该先使用git pull命令拉取较新版本的developing分支并在本地进行合并，手动选择要保留的内容，解决冲突问题之后，再推送到远程仓库。先拉后推。

当同一个项目中的开发人员越来越多时，如果没有制定规则，放任commit就会造成灾难。因此，需要一套流程规则要求团队共同遵守，如Git Flow等。根据Git Flow建议，分支主要分为Master分支、Develop分支、Hotfix分支、Release分支以及Feature分支，各分支负责不同的功能。

Master分支，主要用来存放稳定的项目版本。这个分支的来源只能是从别的分支合并，开发者不会直接commit到这个分支，因为是稳定版本，所以通常会在这个分支的commit上打上版本号标签。

Develop分支，这个分支是所有开发分支中的基础分支。当要新增功能时，所有的Feature分支都是从这个分支划分出去的，而Feature分支的功能完成后，也会合并到这个分支。

Hotfix分支，当在线产品有问题时，会从Master分支划出一个Hotfix分支进行修复，修复好后合并回Master分支，同时合并一份给Develop分支。

Release分支，当认为Develop分支足够成熟时，就可以把Develop分支合并到Release分支，在其中进行上线前的最后测试，测试完成后，Release分支会合并到Master分支和Develop分支。

Feature分支，如果需要新增功能，就要用Feature分支，这个分支是从Develop分支划分出来的，完成后再合并回去。

## 存储仓库管理

### git reflog

```shell
git reflog [show] [ref]
```

该命令用于管理记录到logs目录中的引用日志信息（称为reflogs），包括本地仓库中的分支和其它引用更新的记录。其中，每条记录会持有一个形如ref_name@{#}的名称，数字#指定该条记录是ref_name引用的之前#次时的状态，#从0开始。这种名称可以用于其它命令中。

直接使用该命令git reflot或git reflog show命令，会显示指定引用ref的日志记录，未指定ref时默认显示HEAD头所指向当前分支的日志记录。

```shell
git reflog show
```

```shell
445e3e2 (HEAD -> master) HEAD@{0}: commit: modify main.cpp
5bd30ad HEAD@{1}: commit: create main.cpp
9f7b101 (debug) HEAD@{2}: merge debug: Fast-forward
4e6cecd (origin/master) HEAD@{3}: checkout: moving from debug to master
9f7b101 (debug) HEAD@{4}: commit: debug commit
4e6cecd (origin/master) HEAD@{5}: checkout: moving from master to debug
4e6cecd (origin/master) HEAD@{6}: commit (initial): init commit
```

```shell
git reflog delete ref
```

```shell
git reflog expire --expire=time
```

使用git reflog delete命令删除ref引用的日志记录。使用git reflog expire命令删除所有过期的日志记录，使用--expire选项指定过期时间。

### git clean

```shell
git clean [path]
```

该命令从path所匹配的目录开始，递归删除不在版本控制之下的文件，以清理工作区树。

### git gc

```shell
git gc [option]
```

该命令在当前版本库中执行一些存储管理任务。例如，压缩文件修订（以减少磁盘空间占用并提高性能），移除git add命令创建的不可到达的（unreachable）对象，打包引用，修剪引用日志记录，整理过时的工作区树，更新辅助索引等。

使用--auto选项会检查是否需要并自动执行存储管理任务；使用--aggressive选项以激进方式优化存储仓库，会花费更多时间。

# GitHub

GitHub是一个商业网站，是一个面向开源及私有软件项目的托管平台，其本质是一个Git服务器，只支持Git作为唯一的版本库格式进行托管。

在使用GitHub时，需要拥有一个GitHub账号，然后将通过在GitHub账号中配置SSH Keys或GPG Keys，以允许某个机器连接到该账号。使用如下命令测试SSH密钥是否正常工作。在Linux平台中，Git套件被集成到bash shell终端中；在Windows平台中，可使用git-bash终端。

```shell
ssh -T git@github.com
```

```
Hi Bloonow! You've successfully authenticated, but GitHub does not provide shell access.
```

默认情况下，SSH所使用的本机私钥文件路径为~/.ssh/id_rsa，若所使用的SSH Keys不是该默认路径，可能会报出Permission denied (publickey)错误。可使用ssh-add命令添加指定路径的密钥文件到本机的SSH代理中，使用-l选项查看已添加到ssh-agent中的密钥。

```shell
ssh-add /path/to/key
```

如果遇到Could not open a connection to your authentication agent错误，可能是ssh-agent未启动，使用如下命令启动一个ssh-agent服务。

```shell
eval $(ssh-agent -s)
```

在Windows平台上，如果使用ssh-add /path/to/key命令，可能会遇到如下错误。

```shell
Error connecting to agent: No such file or directory
```

以管理员身份启动PowerShell终端，检查ssh-agent服务是否启动，如下所示。

```powershell
get-service ssh*
```

```powershell
Status   Name               DisplayName
------   ----               -----------
Stopped  ssh-agent          OpenSSH Authentication Agent
```

显示上述信息表示ssh-agent服务为Stopped，即未启动。使用以下两条指令启动ssh-agent服务，其中StartupType选项指定服务启动类型，可选Automatic、AutomaticDelayedStart、Disabled、Manual等值。

```powershell
Set-Service -Name ssh-agent -StartupType Manual
Start-Service ssh-agent
```

此时，启动ssh-agent服务之后，可使用ssh-add命令添加密钥文件。

在配置本机与GitHub建立SSH连接之后，即可将GitHub作为远程存储仓库使用，命令如Git所述。

```shell
echo "Hello World!" > README.md
git init
git add README.md
git commit -m "init commit"
git remote add origin git@github.com:Bloonow/MyLearn.git
git push -u origin master
```

