# 清理右键菜单

首先要使用`Win+R`键入`regedit`打开注册表编辑器。

清除文件夹的右键菜单，在注册表中的`计算机\HKEY_CLASSES_ROOT\Directory\shell\ContextMenuHandlers`下，寻找对应的名称并删除。如果在这里找不到，可以在`计算机\HKEY_CLASSES_ROOT\Directory\shellex\ContextMenuHandlers`、或者在`计算机\HKEY_CLASSES_ROOT\Directory\Background`分支里找到，然后选择删除。

清理右键“新建”菜单，注册表中的`计算机\HKEY_CLASSES_ROOT\.*`根键下存放这所有文件类型的信息，如`计算机\HKEY_CLASSES_ROOT\.docx`就是Word文档的信息，在这个键值下删除`Shellnew`后，右键的“新建”菜单中的“新建Word文档”选项就会被删除。其他类型的文件请注意查对文件类型，然后也可以通过这种方法删除右键“新建”菜单。

# 清理VS Code右键菜单

在VS Code编辑器中，在编辑器Editor界面和资源管理器Explorer界面点击鼠标右键，都会显式上下文菜单（context menu），用于列出快捷命令。

VS Code插件开发者可以通过在插件的\$HOME/.vscode/extensions/NAME/package.json配置文件中添加相应的配置项，详细的配置项见[文档](https://code.visualstudio.com/api/references/contribution-points#contributes.menus)。注意，在服务器上，配置文件可能位于\$HOME/.vscode-server/extensions/NAME/package.json路径。

用于控制编辑器或资源管理上下文菜单的键值，位于package.json配置文件中，位于"contributes":"menus"键下，其名称分别为"editor/context"键与"explorer/context"键。用户可删除其中的值来删除不需要的上下文菜单命令。

# 取消默认应用的打开方式

在注册表管理器中，找到`计算机\HKEY_CLASSES_ROOT\.扩展类型`，删除。再找到`计算机\HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts\.扩展类型`删除。重启计算机就可以了。

# 更改键盘按键的映射

在注册表中，找到`计算机\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Keyboard Layout`，在其Keyboard Layout上右键>新建>二进制值，名称为`Scancode Map`，它的值的格式如下。

<font color="blue">00 00 00 00 00 00 00 00</font> <font color="red">02 00 00 00</font> <font color="gren">35 00 3A 00</font> <font color="purple">00 00 00 00</font>

- 这是一串16进制的数字，每一个数字代表一个16进制位，两个位就代表一个字节。
- 前8个字节（蓝色）表示版本信息，默认16个0。
- 紧跟的4个字节（红色）表示要映射按键的数量，从02开始表示映射一个按键，03表示两个，以此类推；个人推测02表示的是后跟的四字节的个数。
  - 注意，这里采用的是小端模式，即低位在左，高位在右。
- 若要更改n个按键映射，则后跟n个四字节（绿色），这里只修改了一个，就只有一个四字节。
  - 前两个字节表示被映射到的键位的【扫描码+ASCII码】，如这里的35 00中的35就是【/】键的扫描码。此外，【Delete】键的扫描码是0x53。
  - 后两个字节表示要映射按键的【扫描码+ASCII码】，如这里的3A 00中的3A就是【CapsLock】键的扫描码。
  - 这里实现的功能是：将【CapsLock】按键映射为【/】按键。
  - 注：实际执行时，使用ASCII码时映射无效，故都采用了00。
- 最后四字节（紫色），表示二进制结束，默认的8个0。

修改后重启计算机即可生效。

# 删除资源管理器侧边栏条目

在注册表中，找到`计算机\HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\MyComputer\NameSpace`，在该条目下存在若干项，即是资源管理器侧边栏中的条目，对应关系如下表所示。

| 条目                                   | 目录   |
| -------------------------------------- | ------ |
| {0DB7E03F-FC29-4DC6-9020-FF41B59E513A} | 3D对象 |
| {f86fa3ab-70d2-4fc7-9c99-fcbf05467f3a} | 视频   |
| {24ad3ad4-a569-4530-98e1-ab02f9417aa8} | 图片   |
| {d3162b92-9365-467a-956b-92703aca08af} | 文档   |
| {088e3905-0323-4b02-9826-5d99428e115f} | 下载   |
| {3dfdf296-dbec-4fb4-81d1-6a3438bcf4de} | 音乐   |
| {B4BFCC3A-DB2C-424C-B029-7FE99A87C641} | 桌面   |

将不需要的条目直接删除即可。

# 禁止Windows 10自动更新

在设置中，更新和安全>Windos更新>高级选项>更新选项，取消勾选。

在服务services.msc中，找到Windows Update，右键>属性，在常规选项卡中，将启动类型设为禁用；在恢复选项卡中，将三次失败全设为无操作。

在本地策略组gpedit.msc中，计算机配置>管理模板>Windows组件>Windows更新。在配置自动更新项上，右键>编辑，设置为已禁用；在删除使用所有Windows更新功能的访问权限项上，右键>编辑，设为已启用。

在注册表中，找到`计算机\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\UsoSvc`自动更新服务，Start>修改，数值数据为4；FailureActions>修改二进制数据，00000010行与00000018行中，第5个字节位置，由原来的01修改为00。

# CMD启动时执行命令

在Linux平台上，可以在用户家目录下的.bashrc配置文件中指定要在bash shell启动时执行的命令，在Windows平台上也可实现该需求。

创建一个名称为auto_execute_command.bat的批处理文件，编辑文件内容如下所示，下述示例是将CMD的代码页切换到UTF-8字符集。

```cmd
chcp 65001
```

在注册表中，找到`计算机\HKEY_LOCAL_MACHINE\Software\Microsoft\Command Processor`或`计算机\HKEY_CURRENT_USER\Software\Microsoft\Command Processor`，右键>新建>字符串值，名称设为AutoRun，数值指定为批处理文件auto_execute_command.bat的完整路径，即可。
