# Linux系统服务

Linux系统在启动过程中，内核完成初始化以后，由内核启动的第一个程序是/sbin/init程序，进程编号PID为1，以守护进程的方式运行。Linux中所有的进程都由/sbin/init程序进程直接或间接创建并运行的，它通常是系统里所有进程的祖先，负责组织与运行系统的相关初始化工作，让系统进入定义好的运行模式，如命令行模式或图形界面模式。

起初，/sbin/init程序链接到的是sysvinit程序，通过shell脚本以串行的方式启动系统服务，下一个进程必须等待上一个进程启动完成后才能开始启动，因此系统启动的过程比较慢。之后的upstart程序在sysvinit的基础上，把一些没有关联的程序并行启动，以提高启动的速度，但是存在依赖关系的程序仍然为串行启动。而systemd是Linux系统中最新的初始化系统，主要的设计目标是克服sysvinit固有的缺点，提高系统的启动速度，它通过套接字激活的机制，让无论是否有依赖关系的所有程序全部并行启动，并且仅按照系统启动的需要启动相应的服务，最大化提高开机启动速度。

```shell
ls -lhF /sbin/init
```

```shell
lrwxrwxrwx 1 root root 20  3月 24 12:22 /sbin/init -> /lib/systemd/systemd*
```

```shell
pstree -p | grep systemd
```

```shell
systemd(1)-+-ModemManager(933)-+-{ModemManager}(972)
           |-systemd(1553)-+-(sd-pam)(1554)
           |-systemd-journal(379)
           |-systemd-logind(890)
           |-systemd-oomd(710)
           |-systemd-resolve(726)
           |-systemd-timesyn(729)---{systemd-timesyn}(846)
           |-systemd-udevd(430)
```

多数主流Linux系统采用systemd管理系统服务进程，它是服务进程集合的总称，包含负责各种功能的许多进程，用于控制管理系统的资源。systemd的守护进程主要分为系统态（system）与用户态（user），分别位于/lib/systemd目录和/usr/lib/systemd目录中，这两个目录的结构与内容完全一致。

<img src="使用Linux操作系统.assets/systemd目录结构.png" style="zoom: 50%;" />

systemd提供兼容sysvinit的特性，系统中已经存在的服务和进程无需修改；systemd提供比upstart更激进的并行启动能力，更快的启动速度。

systemd提供按需启动的能力，只有在某个服务被真正请求的时候才启动它，当该服务结束时systemd可以关闭它，等待下次需要时再次启动它。

systemd利用Linux内核的特性cgroups来完成跟踪的任务，当停止服务时，通过查询cgroups，systemd可以确保找到所有的相关进程，从而干净地停止服务。

传统的Linux系统中，用户使用/etc/fstab文件维护文件系统的自动挂载点，systemd兼容/etc/fstab文件并管理这些挂载点，以便能够在系统启动时自动挂载它们。同时，systemd内建自动挂载服务，无需另外安装autofs服务，可以直接使用systemd提供的自动挂载管理能力来实现autofs的功能。

systemd自带日志服务journald，该日志服务的设计初衷是克服现有的syslog服务的缺点。systemd-journald用二进制格式保存所有日志信息，用户使用journalctl命令来查看日志信息，无需自己编写复杂的字符串分析处理程序。

![](使用Linux操作系统.assets/systemd架构示例.png)

Linux系统初始化需要做的事情非常多，需要启动后台服务，如启动ssh服务；需要做配置工作，如挂载文件系统。这个过程中的每一步都被systemd抽象为一个配置单元（unit），可以认为一个服务是一个配置单元，一个挂载点是一个配置单元，一个交换分区的配置是一个配置单元等。systemd将配置单元归纳为不同的类型，下面是一些常见的unit类型。

| unit类型  | 描述                                                         |
| --------- | ------------------------------------------------------------ |
| service   | 启动并控制服务后台进程，例如MySQLd，是最常用的一种           |
| socket    | 封装系统中的本地IPC和网络套接字，用于基于套接字的启动        |
| target    | 对其他unit配置单元进行逻辑分组，引用其他配置单元。可以对配置单元做一个统一的控制，例如，将所有图形化服务和配置单元组合为一个target，用于控制系统进入图形化模式 |
| device    | 封装Linux设备树中的设备，用于基于设备的启动                  |
| mount     | 封装文件系统中的一个挂载点                                   |
| automount | 封装文件系统中的一个自动挂载点                               |
| timer     | 定时触发其它unit配置单元                                     |
| swap      | 与挂载配置单元类似，封装内存交换分区                         |
| path      | 当文件系统中的一个文件或目录被修改时启动其它服务             |
| slice     | 对其它用于系统资源管理的配置单元进行分组                     |
| scope     | 与服务配置单元类型，用于管理服务进程                         |

每个配置单元unit都有一个对应的配置文件，并以配置单元的类型为后缀名，位于/usr/lib/systemd/system目录中。系统管理员的任务就是编写和维护这些不同的配置文件，例如一个ssh服务对应一个ssh.service文件，如下所示。

```shell
cat /usr/lib/systemd/system/ssh.service
```

```shell
[Unit]
Description=OpenBSD Secure Shell server
Documentation=man:sshd(8) man:sshd_config(5)
After=network.target auditd.service
ConditionPathExists=!/etc/ssh/sshd_not_to_be_run

[Service]
EnvironmentFile=-/etc/default/ssh
ExecStartPre=/usr/sbin/sshd -t
ExecStart=/usr/sbin/sshd -D $SSHD_OPTS
ExecReload=/usr/sbin/sshd -t
ExecReload=/bin/kill -HUP $MAINPID
KillMode=process
Restart=on-failure
RestartPreventExitStatus=255
Type=notify
RuntimeDirectory=sshd
RuntimeDirectoryMode=0755

[Install]
WantedBy=multi-user.target
Alias=sshd.service
```

Unit段，所有类型的配置单元文件通用，用于定义unit元数据、配置、与其它unit的关系。Description条目描述unit文件信息；Documentation条目描述应用文档；Condition条目描述启动条件；Before条目、After条目、Wants条目、Requires条目用于描述依赖关系。

Install段，所有类型的配置单元文件通用，通常指定运行目标target，使得服务在系统启动时自动运行。Wantedby条目、RequiredBy条目描述依赖关系；Alias条目描述启动运行时使用的别名。

Service段，仅service类型的配置单元文件使用，用于定义服务的具体管理和执行动作。Type条目定义进程行为，如使用fork()启动，使用dbus启动等；ExecStartPre条目、ExecStart条目、ExecStartPos条目、ExecReload条目、ExecStop条目分别指定启动之前、启动时、启动之后、重启时、停止时执行的命令。

作为systemd初始化系统的一部分，可使用systemctl命令管理Linux系统中的进程服务，可使用-t选项指定配置单元unit类型，如service、socket、device等。命令systemctl兼容之前的用于系统服务进程管理的service命令。

```shell
systemctl [option] command [unit]
```

使用systemctl list-units命令列出所有unit信息；使用systemctl status命令查看unit信息；使用systemctl start命令或systemctl stop命令启动或停止一个unit；使用systemctl restart命令或systemctl reload命令重新启动或重新加载一个unit；使用systemctl enable命令启用一个unit随系统启动；使用systemctl disable命令禁用一个unit不随系统启动。

# 配置Linux环境变量

对于Linux平台，可以在.bashrc文件中配置每次bash环境启动是需要执行的命令，并通过export命令将所需的头文件和库文件目录添加到环境变量中。注意，在Linux平台下的.sh文件，要保证其换行符为LF，而不能是Windows平台下的CRLF换行符，否则会出现莫名其妙的的错误。

配置用户bash的环境变量如下所示。推荐在$HOME/.bashrc中添加以下代码，以执行特定的配置文件。

```shell
if [ -f "$HOME/.bash_B" ]; then
    . "$HOME/.bash_B"
fi
```

然后创建$HOME/.bash_B文件，在其中进行用户bash的环境配置，如下所示。

```shell
# Clear firstly
unset C_INCLUDE_PATH CPLUS_INCLUDE_PATH LD_LIBRARY_PATH LD_RUN_PATH

XXX_BIN=$HOME/path/to/xxx/bin
export PATH=$XXX_BIN:$PATH

XXX_INCLUDE=$HOME/path/to/xxx/include
export C_INCLUDE_PATH=$XXX_INCLUDE:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$XXX_INCLUDE:$CPLUS_INCLUDE_PATH

XXX_LID=$HOME/path/to/xxx/lib
export LD_LIBRARY_PATH=$XXX_LID:$LD_LIBRARY_PATH  # during execution
export LIBRARY_PATH=$XXX_LID:$LIBRARY_PATH        # during compling
export LD_RUN_PATH=$XXX_LID:$LD_RUN_PATH          # during linking
```

或者直接在/etc/ld.so.conf文件中，或/etc/ld.so.conf.d/xxx.conf文件中，添加所需库文件的绝对路径（每行一个路径）。然后运行sudo ldconfig命令，以重建/etc/ld.so.cache文件，包含进新添加的路径。

此外，在/etc/profile文件或/etc/profile.d/xxx.sh文件中，配置某个工具或库的二进制可执行文件。

# 图形界面

目前Linux/UNIX中最为流行的两种图形桌面套件是GNOME（GNU Network Object Model Environment）和KDE（Kool Desktop Environment）桌面套件环境。在Ubuntu操作系统中，默认使用的是GNOME桌面套件。

在Ubuntu操作系统的图形界面中，若要进入命令行界面（Command Line Interface，CLI），即纯字符界面，可以使用Ctrl+Alt+F#组合键（F#代表F1,F2,...,F6按键），切换到第x个虚拟字符控制台。Ubuntu操作系统默认开启了六个虚拟终端（虚拟控制台），用于登录到纯字符模式的操作界面，这六个虚拟终端分别表示为tty1,tty2,...,tty6终端。若要从字符控制台返回到已经开启的图形桌面环境，可以使用Alt+F7组合键；而在字符控制台之间进行转换时，只需使用Alt+F#x组合键即可。

> 注意，使用Ctrl+Alt+F#登录到虚拟终端时，并不会关闭图形界面，可使用Alt+F7返回到图形界面。
>
> 注意，若Ubuntu系统在虚拟环境中，某些情况下，组合键中还需加入Shift才能生效。

注意，要执行下述命令，先执行su以进入root账号，避免进入命令行界面时出现卡死的情况。

此外，也可使用如下命令进入或退出纯字符命令行终端界面，如下所示。

```shell
sudo init 3  # 进入命令行终端
sudo init 5  # 退出命令行终端
```

> 注意，使用sudo init 3进入命令行终端时，会关闭图形界面，使用sudo init 5会重新开启图形环境并进入。

下述命令可用于管理图形界面与命令行界面，如下所示。

```shell
sudo systemctl isolate multi-user.target      # 关闭当前图形界面，即启用多用户文本界面，即命令行终端
sudo systemctl isolate graphical.target       # 启动图形界面，即启用多用户文本界面，即命令行终端
```

```shell
sudo systemctl set-default multi-user.target  # 默认命令行终端启动，即机器启动时不启动图形界面
sudo systemctl set-default graphical.target   # 默认图形界面启动
```

```shell
sudo systemctl stop display-manager     # 关闭图形界面
sudo systemctl start display-manager    # 启动图形界面
sudo systemctl disable display-manager  # 禁用图形界面
sudo systemctl enable display-manager   # 启用图形界面
```

# X协议

对于Windows平台来说，图形化界面是在Windows内核中实现的，是操作系统内核的一部分。而对于类Unix系统来说，其内核中并无图形化界面的实现，类Unix系统的图形化界面只是一个应用程序，这些图形化界面的实现底层通常是基于X协议（X protocol）的，也即X协议是类UNIX操作系统用来实现图形界面的，目前是X11版本。

> freedesktop.org以前称为X Desktop Group（XDG），是一个致力于基于X11等桌面环境互操作性和共享基础技术的项目，该项目制定了互操作性规范，并定义了一系列XDG_XXX环境变量，许多工具和应用程序默认使用这些变量。

<img src="使用Linux操作系统.assets/X协议.png" style="zoom:50%;" />

其中，X Server负责管理显示相关的输入输出设备的交互，它负责接受输入设备（键盘鼠标）的动作，并将其告知基于X Client的应用程序，同时负责将图形写入输出设备（显卡等），以进行屏幕画面的绘制与显示。而基于X Client的应用程序，则接受X Server传递的动作事件等，进行程序业务逻辑的处理，并将需要显示的图形告知X Server来显示。

通常来说，X Server与X Client运行在同一主机上，但同时X协议栈也可基于TCP/IP协议，那么X协议就支持X Server与X Client运行在不同主机上。如此，通过SSH X11 Forwarding转发，就可以实现常见的开发场景，即在服务器端运行某个程序（X Client），而在本地显示程序的GUI界面（X Server），这可以满足“在无图形界面的Linux上开发GUI应用程序”的需求。

通过如下命令在Ubuntu系统上安装X11应用程序。

```shell
sudo apt install xorg
```

使用sudo权限修改/etc/ssh/sshd_config文件，打开X11Forwarding和X11UseLocalhost注释，并分别设为yes和no，如下所示。

```shell
X11Forwarding yes
X11UseLocalhost no
```

重启sshd服务，如下所示。

```shell
sudo systemctl restart sshd.service
```

在Windows平台上启用X11 Server有多种方式，这里使用[Xming](http://www.straightrunning.com/XmingNotes/)工具，可在[Public Domain Releases](https://sourceforge.net/projects/xming/files/)网址下载安装。通过所安装的XLaunch启动X11 Server服务，并将Display Number指定为0值，表示采用0.0号进行显示。

打开安装目录下的Xming/X0.hosts文件，将要连接的远程主机IP地址添加到后面新行中，如下所示。

```shell
localhost
10.10.10.102  # Remote Host IP (No this comment)
```

在Visual Studio Code中安装Remote-SSH与Remote X11扩展，配置远程主机的SSH设置如下所示，也即指定X11相关的设置。

```shell
Host MyRemoteHost
  HostName 10.10.10.102  # Remote Host IP (No this comment)
  Port 22
  User MyName
  ForwardX11 yes
  ForwardX11Trusted yes
  ForwardAgent yes
```

在用户的.bashrc文件中，将本地机器的IP地址，以及X11 Server所设置的0.0添加其中，以在每次启动shell时执行，配置bash的环境变量。

```shell
export DISPLAY=10.10.10.101:0.0  # Local Host IP (No this comment)
```

此时，使用VS Code连接远程服务器，执行GUI代码时，即可将图形显示在本地。

# SSH

安全外壳协议（Secure Shell Protocol）是一个网络协议，处于计算机网络协议栈中的应用层，通常基于TCP/IP协议，使用22作为默认端口号。SSH通过在数据传输中使用各种加密技术（例如AES、RSA、ECDSA、SHA-256等算法）来保证通信的安全性，防止数据在网络中被监听、篡改和冒名顶替。

SSH通常用于登录远程计算机的shell或命令行界面（CLI）并在远程服务器上执行命令，它还支持隧道、TCP端口转发和X11连接机制，并且可用于使用关联的SSH文件传输协议（SFTP）或安全复制协议（SCP）传输文件。SSH采用客户端-服务器（Client-Server）架构，SSH客户端程序通常用于建立与SSH守护程序（如sshd）的连接，以接受远程连接。两者通常都存在于大多数现代操作系统中，包括macOS 、多数Linux发行版等。

目前广泛使用的是SSH-2版本，常见的支持SSH协议的软件包括OpenSSH、PuTTY等。OpenSSH套件包含若干工具，将在下面介绍。

在Linux平台上，使用apt install openssh-client命令和apt install openssh-server命令安装OpenSSH客户端与服务器端的相关组件，它们的配置分别位于/etc/ssh/ssh_config文件和/etc/ssh/sshd_config文件中。使用systemctl start ssh命令启动SSH服务，使用systemctl enable ssh命令使SSH随系统启动。

ssh命令连接到指定destination目标主机，并远程登录shell终端，如果指定command命令，则在远程主机执行command命令而不是登录shell终端。其中destination可以是[user@]hostname形式或是ssh://[user@]hostname[:port]形式，其中hostname可以是远程主机IP地址或URL地址。

```shell
ssh [option] destination [command]
```

使用-l选项指定login_name登录用户名；使用-p选项指定port端口；使用-R选项指定address地址；使用-T选项禁止分配虚拟终端；使用-i选项指定验证身份的密钥文件（私钥），默认是位于用户\$HOME/.ssh目录下的id_rsa、id_ecdsa、id_ecdsa_sk、id_ed25519、id_ed25519_sk、id_dsa文件。

```shell
ssh bln@10.10.10.105
```

```
The authenticity of host '10.10.10.105 (10.10.10.105)' can't be established.
ED25519 key fingerprint is SHA256:nH2WQr0seOo+RIkY/yh0UzJ1fHRuxGkzP8S4Am35Tww.
This key is not known by any other names
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added '10.10.10.105' (ED25519) to the list of known hosts.
bln@10.10.10.105's password: 
Welcome to Ubuntu 22.04.3 LTS (GNU/Linux 5.15.0-97-generic x86_64)

  System information as of Tue Apr 16 01:53:12 AM CST 2024

  Memory usage:             6%
  Swap usage:               7%
  IPv4 address for docker0: 172.17.0.1
  IPv4 address for enp4s0:  10.10.10.105
  IPv6 address for enp4s0:  fd00:f484:8de3:2a9b::1011
  IPv6 address for enp4s0:  fd00:f484:8de3:2a9b:67c:16ff:febc:16c9

Last login: Fri Apr 12 11:16:11 2024 from 10.10.10.112
```

ssh-keygen命令生成并管理身份验证密钥文件，默认位于\$HOME/.ssh目录中。在生成过程中会以交互方式确定私钥文件的名称路径，相对的公钥文件以相同路径存储并使用.pub后缀，可选项的密码短语（passphrase），主机密钥必须具有空的密码短语。

```shell
ssh-keygen [option]
```

使用-t选项指定密钥类型，可选rsa、ecdsa、ecdsa-sk、ed25519、ed25519-sk、dsa六种类型，默认使用ed25519类型的密钥；使用-b选项指定所创建密钥的位数；使用-f指定密钥文件的名称路径；使用-s选项指定证书颁发机构的CA密钥文件，以对公钥进行签名；使用-h选项创建主机证书而不是用户证书，以向用户验证服务器主机身份。

ssh-add命令将私钥文件添加到验证代理ssh-agent程序，该命令在使用自定义文件名称路径的密钥时有用。不带参数的版本会加载默认的位于用户\$HOME/.ssh目录下的id_rsa、id_ecdsa、id_ecdsa_sk、id_ed25519、id_ed25519_sk、id_dsa文件。

```shell
ssh-add [option] [path]
```

使用-l选项列出ssh-agent代理当前加载的所有私钥指纹；使用-L选项列出ssh-agent代理当前加载的所有公钥指纹。

ssh-agent是一个持有私钥的程序进程，所持有的私钥用于公钥验证。通过使用环境变量，可以在使用ssh登录其他计算机时找到ssh-agent代理并自动用于身份验证。

sshd是ssh的守护程序进程，监听来自客户端的连接。它为每个请求的连接创建一个新的fork守护进程，以进行密钥交换、加密、身份验证、命令执行和数据交换。sshd可以使用命令行选项或配置文件进行配置，默认配置文件位于/etc/ssh/sshd_config路径。

在sshd的配置文件/etc/ssh/sshd_config中，使用AuthorizedKeysFile条目指定包含公钥的文件，用于公钥认证，默认为用户目录下的\$HOME/.ssh/authorized_keys文件。在\$HOME/.ssh/known_hosts文件中，保存所有已知主机的主机公钥，每当用户连接到未知主机时，其密钥都会添加到每个用户的known_hosts文件中。

使用SSH登录远程Linux服务器时，在用户目录下创建.ssh目录并设置权限。

```shell
mkdir ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

在本地系统中使用ssh-keygen命令创建公钥-私钥对，将本地系统公钥文件id_rsa.pub中的内容拷贝到远程服务器的authorized_keys文件中，即可实现本地系统免密登录远程服务器。

如果使用VS Code登录远程服务器，只需为SSH登录设置指定IdentityFile为本地系统的私钥文件路径即可，如下所示。

```shell
Host connection_name
  HostName remote_host
  Port port
  User user
  IdentityFile $HOME/.ssh/id_rsa
```
