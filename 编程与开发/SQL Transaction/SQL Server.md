> 值得注意的是，本笔记中的一些SQL语句在很大程度上和《MySQL》里的一样，实际上它们都遵守一定的SQL规范，这些SQL语句在一定程度上是通用的。

[TOC]

# 一、概述与补充

本笔记仅为学习记录，可能存在许多错误或者不能包含全部知识，详细请参阅SQL Server帮助文档。

SQL语言根据应用场景可分为：交互式SQL（DBMS）、嵌入式SQL（高级语言）、动态SQL。

SQL语言主要有9个单词引导，表示3种数据库语言。

DDL语句引导词：Create、Alter、Drop，建立、修改、撤销。

DML语句引导词：Insert、Delete、Update、Select，插入、删除、更新、查询。

DCL语句引导词：Grant、Revoke，授权、撤销授权。

注：SQL语言不区分大小写；在以下语法描述中，在`[]`中的内容表示可选项，在`{}`中的内容表示该处为集合中的一个元素，用`...`表示和之前描述类似的若干项。使用的是Microsoft SQL Server。

## （一）数据字典

数据字典（Data dictionary），又称为系统目录（System Catalogs）。它是系统维护的一些表或视图的集合，这些表或视图存储了数据库中各类对象的定义信息，这些对象包括用Create语句定义的表、列、索引、视图、权限、约束等，这些信息又称数据库的元数据，即关于数据的数据。 

不同DBMS术语不一样：数据字典（DataDictionary（Oracle））、目录表（DB2 UDB）、系统目录（INFORMIX）、系统视图（X/Open）。

数据字典也是存储在磁盘上的关系，通常存储模式本身的信息：与关系相关的信息、用户账户信息，包括密码、统计信息与描述信息、物理文件组织信息、索引相关信息。

DBA也可以使用SQL语句来访问数据字典。在高级语言中用SQLDA（SQL Description Area）来使用数据字典，访问数据库中的元数据。

## （二）ODBC、JDBC

ODBC（Open DataBase Connectivity）是一种标准，不同语言的应用程序与不同数据库服务器之间通信的标准。它包含了：

- 一组API，支持应用程序和数据服务器交互；
- 具体DBMS提供一套驱动程序，即Driver库函数（DBMS Driver与数据库服务器通信），Driver被安装到ODBC环境中，共ODBC调用。

JDBC是Java DataBase Connectivity，是一组Java版的应用程序接口API。在JDBC中使用`?`占位符，表示一个宿主程序中的变量，如果有多个占位符，则它们的Index从1开始。

## （三）模式架构

模式Schema。定义模式实际上定义了一个命名空间，在这个空间中可以定义该模式包含的数据库对象，例如基本表、视图、索引等。默认情况下一个用户对应一个命名空间，用户的Schema名等于用户名，并作为该用户缺省Schema；如果没有指定默认架构，则为dbo。又叫做架构（容器），在SQL Server 2005中，一个数据库对象可以通过以下4个命名部分所组成的结构来引用：[服务器].[数据库].[架构].[对象]。

在实际的应用中，架构有几个特定值得注意：

1. 架构定义与用户分开。
2. 在创建数据库用户时，可以指定该用户账号的默认架构。
3. 若没有指定默认架构，则为 dbo，这样是为了向前兼容。
4. 在应用当中，如果通过 uesr.object 来引用一个对象object，会先找与用户默认架构相同的架构（即uesr架构名）下的对象，找不到再找 dbo 对象。

## （四）事务概述

事务是一组数据库操作的序列，这些操作要么一起成功（全做），要么一起失败（全不做）；操作的提交或回退时一同生效的。数据库操作（如INSERT、UPDATE、DELETE）如果是一个事务中的操作，那么要在事务控制之下完成。事务具有4个特性，即ACID特性（ACID properties）。事务操作的数据元素的对象可以是一条记录、一个关系、甚至是一个磁盘块、一个内存页等。

- 原子性（Atomicity）：事务是数据库的逻辑工作单位，事务中包括的诸操作要么都做，要么都不做。
- 一致性（Consistency）：事务执行的结果必须是使数据库从一个一致性状态变到另一个一致性状态。
- 隔离性（Isolation）：并发执行的各个事务的内部操作及其使用数据不能与其他事务相互干扰。
- 持续性（Durability）：一个事务一旦提交，它对数据库中的改变就应该是永久的（即写到磁盘）。

数据库对象的一些关键词可以标识事务，如：`BEGIN TRANSACTION <NAME>`、`COMMIT TRANSACTION <NAME>`、`ROLLBACK TRANSACTION <NAME>`（事务回滚）、`SAVE TRAN`（表示完成部分事务，同时撤销事务的其他部分）。在SQL Server Management Studio中，一个语句默认当作一个事务。

```sql
SET XACT_ABORT ON
BEGIN TRANSACTION T1		-- BEGIN TRAN
	INSERT / UPDATE / DELETE / SELECT / ELSE
COMMIT TRANSACTION T1		-- COMMIT TRAN
```

- `SET XACT_ABORT ON`，当它设为ON时，如果事务语句产生错误将会回滚；默认设为OFF时不会回滚而是继续进行处理，只回滚当前语句；编译错误（如语法错误）不受 XACT_ABORT 的影响。
- 对于大多数的OLE DB提供程序（包括SQL Server），隐式或显式事务中的数据修改语句必须将 XACT_ABORT 设置为ON；唯一不需要该选项的是提供程序支持嵌套事务时。
- 当然也可以手动隐式判断一个事务成功并手动撤销，使用`TRY ... CATCH`，如下`BEGIN TRY BEGIN TRANSACTION ... COMMIT TRANSACTION END TRY BEGIN CATCH ROLLBACK END CATCH`。

两个全局变量`@@ERROR`（如果语句执行成功，该值为0），`@@TRANCOUNT`（记录当前等待提交的事务数）来检测事务的状态。

```sql
SET TRANSACTION READ ONLY 
SET TRANSACTION READ WRITE
```

- 而当事务仅由读或写语句组成，定义此类型即表示随后的事务均是只读型或只写型，直到新的类型定义出现为止。 

# 二、基本SQL语句

## （一）创建

1. **创建一个数据库**

```sql
CREATE DATABASE 数据库名;
```

2. **创建一个表**

```sql
CREATE TABLE 表名 (列名 数据类型 [PRIMARY KEY | UNIQUE] [NOT NULL], ...);
```

- PRIMARY KEY，表示主键约束，在一个表中仅存在一个。
- UNIQUE，表示唯一性约束，是一个表中的候选键。
- NOT NULL，表示一个非空约束，即该列不能取空值。

3. **创建一个视图**

视图包括外层模式+外层模式到逻辑模式之间的映射关系（E-C映像），视图本身没有存储数据，它是存储的定义，使用时再进行映射转换，数据依然存储在基本表中。

```sql
CREATE VIEW 视图名 ([列名, ...]) AS 子查询 [WITH CHECK OPTION];
```

- 使用定义好的视图，可以像使用Table一样，在SQL各种语句中使用，它最终会转换为对基本表的操作。
- 值得注意的是，有些视图映射是不可逆的，如使用了聚集函数或没有主键等的视图，就不能对这一类视图进行更新操作，而只能查询。详细不可更新的视图如下。
- 如果Select目标列中使用了聚集函数。
- Select中使用了Unique或Distinct。
- 使用了Group By。
- 包含由算术表达式计算出来的列。
- 是单个表的列构成，但没有主键。
- SQL视图能进行更新的可执行如：若由单一表构成，且有主键。
- WITH CHECK OPTION，在对视图进行操作时检查是否满足定义的条件。

4. **创建一个索引**

索引是对数据库表中一列或多列的值进行排序的一种结构，使用索引可快速访问数据库表中的特定信息，默认升序，CLUSTERED型（聚簇）索引。聚簇索引的项顺序是和数据页的组织顺序一致的，它的叶子节点就是数据项；而非聚簇不一定，它的叶子节点仅仅是一个数据项存放位置的指针。

```sql
CREATE [UNIQUE] [CLUSTERED] INDEX 索引名 ON 表名(列名 排序方式, ...);
```

5. **创建一个模式**

```sql
CREATE SCHEMA 模式名 AUTHORIZATION 拥有者用户名 [表定义子句 | 视图定义子句 | 授权定义子句];
```

6. **创建一个角色**

```sql
CREATE ROLE 角色名;
```

7. **创建一个用户**

```sql
CREATE USER 用户名;
```

## （二）修改

1. **修改表的定义**

```sql
ALTER TABLE 表名
[ADD {新列名 类型, ...}]; |	-- 增加新列
[DROP [CONSTRAINT] | [COLUMN] {完整性约束名}]; |	-- 删除完整性约束，或列
[ALTER COLUMN {列名 类型, ...}];	-- 修改列的定义
```

2. **修改架构schema**

```sql
ALTER SCHEMA SA TRANSFER SB.TableName | else;
```

- ALTER SCHEMA语句只能用来转换同一个数据库中不同架构之间的对象，架构内部的单个对象可以使用ALTER Table或这View等语句进行更改。该语句把SB架构中的TableName对象转换仅SA架构中。

```sql
ALTER AUTHORIZATION ON SCHEMA ::SA TO User;
```

- 把架构SA的主体改为用户User。

## （三）撤销

1. **撤销表**

```sql
DROP TABLE 表名;
```

- 含表中的数据和表结构本身一并删除；而delete只是删除表中的元组。

2. **撤销数据库**

```sql
DROP DATABASE 数据库名;
```

3. **撤销视图**

```sql
DROP VIEW 视图名;
```

4. **撤销索引**

```sql
DROP INDEX 表名.索引名;
```

## （四）插入

1. **向表中追加元组**

```sql
INSERT INTO 表名 [(列名, ...)] VALUES [(值, ....)];
```

- 值与列名应当保持一致。

2. **批量数据新增命令**

```sql
INSERT INTO 表名 [(列名, ...)] 子查询;
```

- 子查询可以是一个SELECT-FROM-WHERE语句，且SELECT投影的列应该跟INSERT指出的列名一致。
- 需要满足完整性条件。

3. **建表插入**

```sql
SELECT 列名 别名, ... INTO 新表名 FROM 表名 WHERE ... GROUP BY ...;
```

- 在将数据插入到不存在的表时，同时新建该表。

## （五）删除

1. **元组的删除命令**

```sql
DELETE FROM 表名 [WHERE 检索条件];
```

- 若无WHERE则语句删除表中所有元组。

## （六）更新

1. **元组修改命令**

```sql
UPDATE 表名 SET 列名 = 值表达式 | 子查询的结果, ... [WHERE 检索条件];
```

## （七）查询

### 1. 单表查询

```sql
SELECT 列名, ... FROM 表名 WHERE 检索条件;
```

- 相当于∏~列名,...~(σ~检索条件~(表名))，关系代数语句。
  
- select、from、where被称为子句。
  
- SELECT * 表示所有信息列。
  
- 集合要元素求唯一，但DBMS实际应用时允许重复，且单独投影一个属性也能重复。
  
- 可以对列名进行操作，如SELECT 2020 - Page AS Birth。
  
- 检索条件可以使用`[NOT] BETWEEN xxx AND xxx`代替 xxx <= y and y >= xxx。

1. **保留一份查询结果**

```sql
SELECT DISTINCT 列名, ......
```

- 当查询结果有相同的两个或多个时，只保留一份结果。

- 缺省时使用的是ALL关键字。

- 如果DISTINCT后跟的是多个列名，则保留根据是多个列组成的整体的唯一性。

2. **对查询结果排序**

```sql
SELECT ... FROM ... WHERE ... ORDER BY 列名 [ASC | DESC];
```

- ASC表示根据所选列名升序排序。
- DESC表示根据所选列名降序排序。

3. **模糊查询**

```sql
... WHERE 列名 [NOT] LIKE '字符串模板';
```

- 其中有`%`通配0或多个字符，`_`匹配一个字符，`\`转义字符，和正则表达式如`[a-z]`。
- 值得注意的是，在早期版本，一个汉字通常是两个字符，注意编码格式；现在一般一个汉字就是一个字符，可以使用一个`_`查找出来。
- SQL需要自己指定转移字符，形式如下：like '%\\_%' ESCAPE '\_' ，表示匹配带 _ 的数据。
- 更简单的可以使用 [ ]，正则表达式的写法，如：like '%[_]%' 也可以。

### 2. 多表查询

```sql
SELECT 列名, ... FROM 表名1, 表名2, ... WHERE 检索条件;
```

- 相当于∏~列名,...~(σ~检索条件~(表名1 × 表名2 × ...))，关系代数语句，含义是各个表之间的笛卡尔积，而不是链接。其中用σ引出的索引条件，就相当与θ连接中的θ。

- 若θ连接中的等值连接，有WHERE 表1.属性A = 表2.属性A。

1. **别名**

表的别名实现自连接，同时列也可以指定别名：

```sql
表名 [AS] 表别名, ...
列名 [AS] 列别名, ...
```

如：SELECT S1.name, S2.name FROM Student S1, Student AS S2 WHERE S1.score > S2.scroe;

### 3. SQL复杂查询

使用In、Not In、θ some、θ all、Exists、Not Exists、聚集函数、Group by、Having。其中In、Not In、θ some、θ all都能被Exists、Not Exists所表示。

子查询：出现在WHERE语句中的内部Select-From-Where被称为子查询。它返回了一个集合，与这个集合进行比较可以确定另一个查询（外部查询）的集合。

1. **集合成员资格**

```sql
... WHERE 表达式 [NOT] IN (子查询);
```

- 判断某一表达式的值是否存在子查询的结果中，子程序可以直接指定一个已知的集合，如('张三', '李四')。
- 注意集合各项与表达式的对应。
- 子查询为内层，外面的为外层。若内层不引用外层符号，则称为非相关子查询；若内层使用外层符号（即外层向内层传递参量），需要用外层表名或表别名形式来限定，称为相关子查询。

2. **集合之间比较**

```sql
表达式 θ SOME (子查询);	-- SOME 等价于 ANY，但意义上有歧义，不推荐使用
表达式 θ ALL (子查询);
```

- θ是六种比较运算符：<、<=、>、>=、=、!=，不等于也可以使用<>。
- θ SOME，表示表达式与子查询中的某一个符合θ关系即可，存在。
- θ ALL，表示表达式与子查询集合中的所有全部满足θ关系，任意。

3. **集合基数的测试**

```sql
[NOT] EXISTS (子查询);
```

- 判读子查询中有无元组存在。
- 只关心子查询中的集合中是否有元素，如：EXISTS 空子查询，为false；NOT EXISTS 空子查询，为true。
- Exists可以直接不用，而使用外层查询表示。
- Not Exists常用于双重否定或正面不好求解的情况。SQL中没有全程量词，可以将其转换为存在量词的形式。$\forall(x)P \equiv \neg(\exists x(\neg P))$。
- 一般而言为相关子查询。EXISTS语句后跟的子查询用到外层查询的变量。

### 4. 分组与函数

1. **结果计算**

```mysql
SELECT 列名 | expr | agfunc(列名), ... FROM ... WHERE ...;
```

- expr可以是常量、列名或常量列名、特殊函数以及算术运算符构成的算术运算式。
- agfunc()是一些聚集函数。SQL提供了5个标准的内置函数COUNT、SUM、AVG、MAX、MIN。
- 聚集函数是不运行用于Where子句的，Where子句是对每一个元组进行条件过滤；而聚集函数是对集合（组）进行操作。

2. **聚集函数**

```sql
COUNT(*)
COUNT(ALL | DISTINCT 列名满足条件)
```

3. **分组**

```sql
SELECT ... FROM ... WHERE ... GROUP BY 分组条件;
```

- 分组条件可以为：列名1, 列名2, ... ，表示按照列名，将这些列上的值都相同的元组作为一组，返回的是一些集合（组），每组是一个元组的组合。
- 具有相同条件值的元组划到一个组或一个集合中，同时处理多个组的聚合运算。
- 注意，出现GROUP BY语句时，不允许在SELECT子句中出现包含其他字段的表达式。
- 一旦出现了分组GROUP BY子句，则集合函数是对组内进行操作。

4. **分组过滤**

```sql
SELECT ... FROM ... WHERE ... GROUP BY 分组条件 [HAVING 分组过滤条件];
```

- 保留满足条件的分组，不满足的过滤掉。
- 先用Where找出所有元组，用Group by分组，再用Having过滤。

### 5. 基于派生表的查询

将FROM或WHERE后的子查询作为一个临时表，来进行查询。如：
```sql
FROM (SELECT Sid, AVG(Sscore) FROM Students GROUP BY Sid) AS TempTable(Sid, Savgrage)
```



## （八）关系代数运算

### 1. 并、交、差

```sql
子查询1 UNION [ALL] | INTERSECT [ALL] | EXCEPT [ALL] 子查询2
```

- 设结果元组在子查询1中重复出现m次，在子查询2中重复出现n次。
- 在通常情况，即不适用ALL关键字时，做的是集合的运算，删除在子查询中重复出现多次的元组，保留唯一。
- 如果使用ALL关键字，则做的是包的运算，并交差分别保留m+n、min{m,n}、max{0,m-n}次。
- Intersect、Except并没有增加SQL的表达能力，它们可以用其它语法代替。
- 有的DBMS可能不支持。

### 2. 检验空值

```sql
... WHERE 列名 IS [NOT] NULL
```

检测指定列的值是否为空。先行DBMS中空值处理如下。值得注意的是，当子查询返回一个空值时，这时候返回结果不是空集，而是一个仅有一个元素的集合，这个集合中的唯一元素是空值。

- 除`Is [Not] Null`外，空值不能满足任何查询条件。
- 如果Null参与算术运算，则该算术表达式的值为Null。
- 如果Null参与比较运算，则该结果为FALSE。在SQL-92标准中为Unknown。
- 如果是 expr θ ALL 子查询 语句种的子查询返回空集（不是空值）时，无论比较谓词是什么，整个式子是TRUE。
- 如果Null参与聚集运算，则除`COUNT(*)`之外，其他聚集运算都忽略有Null的元组。
- 使用排序`ORDER BY`的情况，NULL值被当作最小值处理。
- 在使用`DISTINCT`、`GROUP BY`的情况下，空值NULL被看作相同，即一个取值处理，从而只保留一个或只形成一个分组。
- 两个表进行连接时，R.s = T.s条件下，不会将同时空值的元组选出来。

有一个函数，`ISNULL(属性列, 0)`，表如果属性列的值是空值，则将他应用成0（不更改原表）。

### 3. 连接操作

```sql
... FROM 表1 [NATURAL] 连接类型 JOIN 表2 连接条件 ...
```

- 连接类型有`INTER`、`LEFT OUTER`、`RIGHT OUTER`、`FULL OUTER`，分别表示θ连接、左外连接、右外连接、全外连接。
- 使用`NATURAL JOIN`表示自然连接。
- 连接条件有：`ON 连接条件`，表示θ连接的θ条件；`USING 列名, ...`，指定两个表的某些列名要取相同值，且只出现一次。



# 三、嵌入式SQL语言

将SQL语言嵌入到某一个高级语言（称为宿主语言，Host Language），这里例子的宿主语言是C语言。在C语言中嵌入式SQL中，为了能够快速区分SQL语句与宿主语言，所有SQL语句都必须加前缀EXEC。有一些典型的特点，如exec sql引导语句，into :变量名，可以接受SQL语句检索结果等。

宿主语言程序、嵌入式SQL语句、Database之间一般有如下几个问题需要考虑。

1. 宿主程序如何与数据库连接和断开连接。
2. 如何将宿主语言的变量传递给SQL语句。
3. SQL语句如何执行。
4. 如何将SQL检索到的结果返回给宿主程序。
5. 宿主程序如何知道SQL语句的运行状态，成功或异常，错误捕获等。
6. 静态SQL语句中常量更换为变量。
7. 动态SQL，根据条件构造SQL语句，编程者对所需字段已知或未知。

## （一）程序与数据库的连接和断开

连接和断开的语法根据不同的软件产品会有所不同，这里用SQL标准中建议的。

连接和断开分别如下：

```c
EXEC SQL CONNECT TO target-server AS connect-name USER "username"/"password";
EXEC SQL CONNECT TO DEFAULT;
EXEC SQL DISCONNECT connect-name;
EXEC SQL DISCONNECT CURRENT;
```

## （二）事务处理

在SQL语句执行过程中，必须有提交和撤销才能确认其操作结果。在嵌入式SQL语句中，只要该程序当前没有正在处理的事务，任何一条数据库语句（如EXEC SQL SELECT）都会引发一个新事务的开始；而事务的结束是需要应用程序通过commit或rollback确认的；因此Begin Transaction和End Transaction两行语句不是必须的。

```c
[BEGIN TRANSACTION]
	EXEC SQL ...;
	/* xxx */
	EXEC SQL COMMIT WORK;	// 提交
	EXEC SQL COMMIT RELEASE;	// 提交并断开连接
	EXEC SQL ROLLBACK WORK;	// 撤销
	EXEC SQL ROLLBACK RELEASE;	// 撤销并断开连接
[END TRANSACTION]
```

## （三）在嵌入式SQL中使用宿主变量

在一个变量名之前加上冒号，表示它式宿主程序的变量，如`:变量名`。

这些变量要特殊声明：

```c
EXEC SQL BEGIN DECLARE SECTION;
	char vName[] = "张三";
EXEC SQL END DECLARE SECTION;
```

- 有的DBMS支持宿主程序中的数据类型与数据库的类型之间自动转换，有的不支持数据类型的自动转换。

## （四）查询

1. **检索单行**

```c
EXEC SQL SELECT 列名, ... INTO :变量名, ... FROM 表名 | 视图名 WHERE 检索条件;
```

2. **用游标读取多行数据（元组）**

使用游标Cursor要先定义、再打开（执行）、接着一条一条处理，最后再关闭。游标是系统为用户开设的一个数据缓冲区，存放SQL语句的执行结果，每个游标区都有一个名字。用户可以通过游标逐一获取记录并赋值给变量，交给宿主语言进一步处理。

```c
EXEC SQL DECLARE 游标名 CURSOR FOR SELECT ... FROM ... WHERE ...;
EXEC SQL OPEN 游标名;
EXEC SQL FETCH 游标名 INTO :变量名;
EXEC SQL CLOSE 游标名;
EXEC SQL DEALLOCATE 游标名;
```

3. **可滚动游标**

标准游标始终式从开始到结束移动，一次一条记录。

ODBC（Open DataBase Connectivity，开放数据库互联）式一种跨DBMS操作DB的平台，它在应用程序与实际DBMS之间提供了一种通用接口。许多实际的DBMS不支持可滚动的游标，但可以ODBC实现此功能。

```c
EXEC SQL DECLARE 游标名 [INSENSITIVE] [SCROLL] CURSOR [WITH HOLD] FOR 子查询 [ORDER BY 结果列[ASC | DESC], ...] [FOR READ ONLY | FOR UPDATE OF 列, ...];
EXEC SQL FETCH [NEXT | PRIOR | FIRST | LAST | [ABSOLUTE | RELATIVE] 决定或相对偏移值] FROM 游标名 INTO :宿主变量, ...;
```

- Scroll表示该游标是可滚动的。
- 相对偏移值，正值表示向结束方向滚动，负值表示向开始方向滚动。
- EOF表示结束，BOF表示开始。
- 子查询的Where语句的条件可以写成 CURRENT OF 游标名。

```assembly
CLOSE 游标名
DEALLOCATE 游标名
```

- 删除和释放游标。

## （五）维护操作

1. **删除**

可以使用查询删除或定位删除两种方式。

```c
EXEC SQL DELETE FROM 表名 { WHERE 查询条件 | WHERE CURRENT OF 游标名 };
```

2. **更新**

可以使用查询更新或定位更新两种方式。

```c
EXEC SQL UPDATE 表名 SET 列名 = 值或表达式, ... { WHERE 查询条件 | WHERE CURRENT OF 游标名 };
```

3. **插入**

```c
EXEC SQL INSERT INTO 表名 [(列名, ...)] [VALUES (值或表达式, ...) | 子查询];
```

## （六）嵌入式SQL执行状态

如何捕获嵌入式SQL语句的运行状态及其处理。

1. **设置SQL通信区**

```c
EXEC SQL INCLUDE SQLCA;
```

2. **设置状态捕获语句**

```c
EXEC SQL WHENEVER 捕获条件 动作;
```

- 该语句所设置条件的作用域从这条Whenever开始，直到下一条Whenever语句结束。对Whenever可能引起死循环，在处理的动作中或所需位置一般加上EXEC SQL WHENEVER CONTINUE语句。
- 它会对在作用域内的所有Exec Sql语句所引起的对数据库的调用自动检查是否满足捕获条件，如果满足则执行动作。
- 条件有：SQLERROR语句错误，NOTFOUND无查询结果，SQLWARNING警告。
- 动作一般可为：continue继续执行，goto 标号，stop终止，函数调用或其他。

3. **状态记录**

典型DBMS状态记录的三种方法：

1. SQLCODE，不同DBMS中定义的值表示不同的含义，一般等于0表示正常，小于0表示错误，大于0表示警告。
2. SQLCA.SQLCODE，一般SQLCODE在通信区中。
3. SQLSTATE状态信息。

## （七）嵌入SQL动态语句

动态构造SQL语句的字符串，然后交给DBMS执行，注意高级语言字符串中一些字符的转义，推荐使用字符串的生模式。

1. **立即执行动态SQL语句**

```c
EXEC SQL EXECUTE IMMEDIATE :字符串名;
```

- 字符串名是宿主语言中的一个字符串变量，它的内容是一个SQL语句。该SQL语句中没有“变量”参数。

2. **Prepare-Execute-Using语句**

Prepare预编译，编译后的SQL语句允许动态参数，Execute语句执行，用Using语句将动态参数传递给编译好的SQL语句。

```c
EXEC SQL PREPARE sql_temp FROM :字符串变量名;		// 如，sql_text = "select * from :table_name"
EXEC SQL EXECUTE sql_temp USING :变量名;	// 如，vTable = "Student";
// 上述相当于，EXEC SQL EXECUTE IMMEDIATE "select * from Student";
```

# 四、过程化SQL

SQL 99标准支持过程和函数的概念，SQL可以使用程序设计语言来定义过程和函数，也可以用关系数据库管理系统（交互软件）自己的过程语言来定义。如Microsoft SQL Server的Transact-SQL、Oracle的PL/SQL等都是过程化SQL的编程语言。

SQL Server的过程化SQL是指在 SQL Server Manager Studio 的交互页面进行操作的，它的基本形式和嵌入式SQL类似；而且在SQL中也提供了一些流程控制语句的支持，如：`declare @variable 类型`、`while`、`begin ... end`、`if`。其中使用一个`@`表示是一个变量，使用两个`@@`表示的引用全局变量。比如`@@fetch_status`是在游标查询后的状态，为0时表示 fetch 成功，有数据。

```sql
declare @i int		-- 声明变量
set @i=1			-- 赋初值
declare c_sc cursor for		-- 声明一个游标
	select * from sc order by grade desc	-- 按成绩降序
open c_sc			-- 打开，即执行游标
fetch c_sc			-- 获取当前游标的元组
while (@@fetch_status=0)
begin
	update sc set n=@i where current of c_sc	-- 更新排名
	set @i=@i+1
	fetch c_sc
end
close c_sc			-- 关闭游标
deallocate c_sc		-- 撤销分配的空间
```

上述用法是在客户端编程，如果想要把它保存到服务器中，可以将它创建为过程，保存到服务器上。`CREATE PROCEDURE 存储过程名 AS BEGIN ... END;`，它保存在服务器的数据库中的存储过程中，可以由不同的用户使用。

## （一）存储过程

SQL Server 为 Transact_SQL 提供了许多操作的一系列集合，并将它们做成一个语句，称为存储过程。

- `sp_password old, new, login`，该存储过程可以修改密码，old是原密码，可以为空，用于在忘记密码时指定新密码。
- `sp_addlogin loginName, password, database, language, sid, encryption_option`，创建新的使用SQL Server认证模式的登录账号。
- `sp_droplogin login`，删除登录账号，禁止其访问SQL Server。
- `sp_grantdbaccess login, name_in_db`，为登录名建立一个数据库的用户名，注意是在哪个数据库下操作的。
- `sp_revokedbaccess name`，将数据库用户从当前数据库中删除，其匹配的登录者就无法使用该数据库。
- `sp_helptext 存储过程名`，显示一个存储过程的语句。
- `sp_rename 'old_name', 'new_name'`，更改对象名。
- `sp_addtype typename 'aType'`，创建一个类型。
- `sp_help obj`，显式数据库对象的信息。
- `sp_lock Sql-Transacts`，观察事务运行过程中各个阶段的锁状态。

也可以自己创建一个存储过程，基本语法如下，详细参考联机帮助。

```sql
CREATE PROCEDURE 存储过程名 @参数 [AS] 类型 [= default], ... AS BEGIN
... -- 一个操作或者其他过程
END
```

## （二）函数

SQL Server中提供了一系列函数，如`substring(col, start, len)`取一个字符串列属性的子字符串。

SQL Server提供了许多内置函数，如字符处理函数、聚集函数、统计函数、数值函数等等。当然也可以自定义函数，基本语法如下，详细请参考联机帮助。

```sql
CREATE FUNCTION [模式名.] 函数名 @参数 [AS] 类型 [= defualt], ...
RETURN 返回类型, ... AS BEGIN
... -- 函数体
RETURN 返回值
END
```

一个例子：

```sql
CREATE FUNCTION f_count(@sno char(8)) RETURN INT
AS BEGIN
	DECLARE @i INT;
	SELECT @i=count(*) FROM SC WHERE Sno = @sno;
	return @i;
END
```

# 五、安全性与完整性

有的DBMS能操纵多个数据库，指定当前数据库，和关闭当前数据库分别为：

```sql
USE 数据库名;
CLOSE 数据库名;
```

## （一）安全性约束

安全性约束包括以下分类：

1. 自主安全性机制，存取控制，通过用户将权限传递，使用户自主管理数据库的安全性。
2. 强制安全性机制，强制分类，使不同级别的用户访问到不同类别的数据，TS、S、C、P。
3. 推断控制机制，防止通过历史信息，推断出不该被知道的信息；防止通过公开信息推断出个人隐私信息。
4. 数据加密机制，密钥、加密/解密方法与传输等。

用户可分3类：超级用户（DBA）> 账户级别（程序员用户）> 关系级别（普通用户）。

权限可分3级别：级别1，读；级别2，更新；级别3，创建。高级别自动包含低级别。

安全性控制规则/访问规则，Access Rule ::= (S, O, T, P)，表示了S在P的条件下使用T对O的访问。Access Rule的集合通常存放在数据库字典或系统根目录中，构成了所有用户的访问权限。

- S，请求主体（用户）
- O，访问对象，表、数据项等，粒度可大可小
- T，访问权利，增删改查创等
- P， 谓词，访问条件

自主安全性机制可以通过存储矩阵、视图等手段实现，视图机制、审计、数据加密等。

### 1. 授权命令

```sql
GRANT { ALL PRIVILEGES | 权限, ... } ON 对象类型 对象名, ... TO { PUBLIC | 用户ID, .... } [WITH GRANT OPTION];
```

- 对象类型和对象名可以是数据库、TABLE 表名、VIEW 视图名、列名等。在SQL Server中不用指定对象类型。
- 权限有Select、Insert、Delete、All Privileges，如 UPDATE(某列) 。有的数据库软件可以授予用户修改表结构的权限`ALTER`，有的数据库没有，可以授予用户创建表的权限`CREATE TABLE`，它包含修改表结构的权限。
- `Public`表示所有用户。
- 用户ID表示某一个用户账户，是由DBA创建的合法账户，也可以是一个角色。
- WITH GRANT OPTION，表示允许被授权者传播这些权利。

给一个角色授权的方式类似给用户授权，如下：

```sql
GRANT { ALL PRIVILEGES | 权限, ... } ON 对象类型 对象名, ... TO 角色名1, ...;
```

把一个角色的权限赋给某个用户，如下：

```sql
GRANT 角色1, ... TO 用户或其他角色, ... [WITH ADMIN OPTION];
```

- WITH ADMIN OPTION，指定被授权的角色或用户可以给其他用户二次授权。
- 被授权的用户，拥有所有角色的权限。

### 2. 撤销授权

```sql
REVOKE { ALL PRIVILEGES | 权限, ... } ON 对象类型 对象名, ... FROM { PUBLIC | 用户ID, .... } [CASCADE | RESTRICT];
```

- 对象类型和对象名可以是 TABLE 表名、VIEW 视图名、列名等。
- 使用CASCADE（有的数据库为RESTRICT），将自动执行级联操作，级联收回由该用户分发的权限。

收回一个角色的权限类似：

```sql
REVOKE 权限 ON 对象名 FROM 角色;
```

除此之外，当一组用户属于某个角色，对这个角色进行授权，但又要收回某一个用户的权限，使用REVOKE是解决不了的，这时就要使用DENY。

```sql
DENY 权限 ON 对象名 TO 用户ID;
```

### 3. SQL Server用户权限等

权限有CREATE、SELECT、INSERT、DELETE等，SQL-Transaction语法就是使用了GRANT、REVOKE等语法。

可以使用图形化界面，在数据库服务实例上，有一个安全性（Security）选项，其子级别下有一个Login项，它是登录名，针对系统的，可以在其中创建登录实例的账户等。注意需要先设置实例的Security成为windows+sql server身份验证。

每个数据库有一套自己的权限系统，在数据库级别下，也有一个安全性选项，其子级别下有一个用户（Users）项，其中单独记录了该数据库授权的用户。还有一个角色（Roles）项，它是一组特定的用户，为了授权方便；一个用户如果是哪个角色，就可以使用角色所拥有的权限。还有一个模式（Schemas）项，它之中记录了某个模式，一个模式是属于某一个用户Uesr的。

实例服务的Login和具体数据库的Users不一定完全一样，它们之间存在一种类似映射对应的关系。一个登录名在一个数据库中对应一个用户名，一个登录名可以对应多个数据库中的用户名。

一些举例如下：

- `sa`，用户，系统管理员（system administrator），具有所有权限。
- `dbo`，用户，数据库所有者。
- `guest`，角色，访客。
- `public`，角色，包含所有的用户。

### 4. 强制安全性机制MAC

被分为主体（用户、进程等）和客体（基本表、文件等）。主体和客体都有敏感度标记，分别叫许可证级别和密级。有以下规则：

主体级别高的可以读取客体级别低的数据。

主体级别低的才可以写入客体级别高的数据。

## （二）完整性约束

完整性约束按不同的分类标准可以有：广义完整性，狭义完整性；域完整性，关系/表完整性；结构完整性，内容完整性；静态完整性，动态完整性。

完整性约束的一般形式，Integrity Constraint ::= (O, P, A, R)

- O，数据集合，约束的对象
- P，谓词条件，什么样的约束
- A，触发条件，什么时候触发
- P，响应动作，不满足怎么办

### 1. SQL静态约束

用Create Table创建表的时候指定、Alter修改。

```sql
CREATE TABLE 表名 (列名 类型 [DEFAULT {默认常量 | NULL}] 列约束, ..., 表约束, ...);
```

其中一个列约束的基本形式如下：

```sql
{ NOT NULL | [CONSTRAINT 约束名] 
	{ UNIQUE | PRIMARY KEY | CHECK(条件表达式) | 
	REFERENCES 另一表名[(另一表主键)] [ON DELETE | ON UPDATE { CASCADE | SET NULL | NO ACTION }] 
	}
}
```

- 其中Unique指定唯一性。
- Primary Key指定主键；FOREIGN KEY(列名)指定外键。
- Check的条件表达式可以是一个Select-From-Where，也可以是对列值的约束，来表示用户自定义完整性。
- References表示外键，它引用另一个表的主键，如果外键和主键同名，则另一表主键可省略。它之后的On Delete表示当另一表中，一个元组被删除（主键被删除）后，该表中外键与之对应的元组如何办。Cascade表示删除该表中与之对应的元组，Set Null表示被删除的对应外键置空。除此之外，还有受限删除（RESTRICTED）、置默认值删除（SET DEFAULT）、拒绝执行（NO ACTION）等，插入时有受限插入、递归插入等。
- 当然对于References引用外键也有对ON UPDATE的操作的指定行为，一般与ON DELETE相同。

一个表约束的基本形式如下：

```sql
[CONSTRAINT 约束名] { UNIQUE(列名, ...) | PRIMARY KEY(列名, ...) | CHECK(条件) |
FOREIGN KEY 列名, ... REFERENCES 另一个表名(另一表列名, ...) [ON DELETE | ON UPDATE CASCADE] }
```

- 表约束中的各可选项等，跟列约束中的类似。
- Check的条件，应该是一个元组多个列值满足的条件，而不是同一列的多个元组。条件中只能使用同一个元组的不同列的当前值。

值得注意的是，如果一个表自参照（即自己的某属性参照自己的另一属性），可能是无法定义的，解决方法是先用 create table 创建一个主键约束，再用 alter table 修改外键约束；也可能会容易造成无法启动的情况，系统同通过事务完毕后再检查。

用**Alter**来撤销或追加约束。

```sql
ALTER TABLE 表名 [ADD (列名 类型 可选默认 列约束)] [DROP { COLUMN 列名 | 列名, ...}] [MODIFY (列名, ...)] [ADD CONSTRAINTS 约束名] [DROP CONSTRAINTS 约束名] [DROP PRIMARY KEY];
```

- 不同的DBMS实现可能有所不同。如下一个例子。

```sql
ALTER TABLE Orders ADD CONSTRAINT FK_Products
	FOREIGN KEY (product_id) REFERENCES Products(prod_id) ON DELETE NO ACTION;
```

### 2. 用户自定义完整性

SQL中用于属于约束性方面的有`NOT NULL`、`CHECK`子句；而用于全局约束方面的有`CREATE ASSERTION`、`CREATE RULE`等语句。

1. **域中的完整性约束**

一般地，域是一组具有相同数据类型的值的集合。SQL支持域的概念，并可以用`CREATE DOMAIN`语句创建一个域以及该域应该满足的完整性约束条件，然后就可以用这个自定义域来定义属性。优点是，数据库中不同的属性可以来自同一个域，当域上的完整性约束条件改变时，只需要修改域的定义，而不必一个一个地修改使用这个类型的每个表。

```sql
CREATE DOMAIN 自定义域名 基本类型 约束;	-- Or
exec sp_addtype TypeName 'aType';	-- a type like DECIMAL(10,2) and etc.
```

- 自定义域的完整性约束，也可以使用DROP、UPDATE等语句修改等，自定义域名和基本数据类型的使用方法一样。

也可以先创建一个域，再创建一个规则（CREATE RULE），然后用存储过程（sp_bindrule）绑定，如下：

```sql
CREATE RULE ruleName AS condition_expression;
sp_bindrule [@rulename=] 'ruleName', [@objname=] 'object_name' [,'futrueonly'];
sp_unbindrule [@objname=] 'object_name' [,'futrueonly'];  -- 解绑定
```

- condition_expression包含一个变量，每个局部变量前都有一个`@`符号。该表达式引用通过UPDATE或是INSERT语句输入的值。
- futrueonly，用在用户自动类型上，表明该约束仅对之后再使用用户类型的列应用该Rule，而已存在使用的则不应用该Rule。
- 一般可以绑定到某一列，或者是用户自定义的数据类型。值得注意的是，在绑定RULE时对于表中已经存在的数据不起作用。

2. **使用断言来定义约束**

```sql
CREATE ASSERTION 断言名/约束名 CHECK(谓词条件);
```

- 谓词条件可以是一个条件表达式，或者一个Select-From-Where等。
- 表约束和列约束就一一些特殊的断言。
- 使用断言来定义约束，在每次更新是，DBMS自动检查断言。但过多断言会降低效率，故谨慎使用。

### 3. 触发器实现动态完整性约束

触发器Trigger，在SQL表示一段可以在特定时刻被执行的程序。它存储在服务器中，可以看成是一段自动激活的存储过程。注意理解 SQL-Transactions 语句的执行和提交是不同的时刻或者状态。

一个触发器只适用于一个表，每个表最多只能有三个触发器，分别是INSERT、UPDATE、DELETE触发器。每个触发器都有两个特殊的表，即插入表（INSERTED）和删除表（DELETED），这两个表是逻辑表，并且是由系统管理的，它们的结构总是与被该触发器作用的表有相同的表结构；它们存储在内存中，不是存储在数据库中，因此不允许直接对其修改。

- INSERT触发器：先向INSERTED表中插入一个新行的副本，然后检查INSERTED表中的新行是否有效，确定是否要阻止该插入操作；如果所插入的行中的值是有效的，则将该行插入到实际基本表中。
- UPDATE触发器：先将原始数据行移到DELETED表中，然后将一个新行插入到INSERTED表中，最后计算DELETED表和INSERTED表中的值以确定是否干预。
- DELETE触发器：将原始数据行移动到DELETED表中，计算DELETED表中的值决定是否进行干预，如果不进行，那么把该行数据删除。

值得注意的是，不同的DBMS产品对插入表、删除表（SQL Server中）的定义不同，在Oracle中与之相应的是NEW和OLD表。

1. **可能在Oracle中的创建语句**

```sql
CREATE TRIGGER 触发器名 BEFORE | AFTER { INSERT | DELETE | UPDATE [OF 列名, ...] }
ON 表名 [REFERENCING 变量名, ...] [FOR EACH ROW | STATEMENT] [WHEN(检测条件cond)]
{ 语句 | BEGIN [ATOMIC] 语句集 END }
```

- 意义：当某一个事件发生时（Before | After），对该事件产生的结果（或是一个元组，或是整个操作的所有元组），进行条件检查cond，如果满足条件，则执行后面的程序段。在检查条件或程序中引用的变量可用变量名指定。
- SQL Serve中使用的是`CREATE TRIGGER 触发器名 ON 表名 FOR 操作如update AS BEGIN ... END`。
- 变量名可为：`NEW | OLD [ROW] [AS] 变量名`，表示元组；`NEW | OLD TABLE [AS] 变量名`，表示表。上述是Oracle数据库语法；在SQL Server中使用的是 INSERTED 和 DELETED 对应 NEW 和 OLD。
- NEW / OLD、INSERTED / DELETED 可以为一个表，如 SELECT ... FROM INSERTED，它和所操作的表的属性列一样。
- For Each Row表示对每一个元组检查When条件，For Each Statement表示对当前操作的所有元组检查When条件；即行级触发和全部触发，在SQL Server中不支持。
- Atomic表示原子操作，它指定在它之后的语句集为原子性。

一个例子，进行Teacher表更新元组时，使其工资只能升不能降。

```sql
CREATE TRIGGER teacher_chgsal BEFORE UPDATE OF salary
	ON Theacher
	REFERENCING NEW x, OLD y
	FOR EACH ROW WHEN (x.salary < y.salary)
	BEGIN
	raise_application_error(-20003, 'invalid salary on updata');
	END;
```

2. **在SQL Server中的创建语句**

```sql
CREATE TRIGGER <触发器> ON <表名|视图名> [WITH ENCRYPTION] {FOR | AFTER | INSTEAD OF}
{[DELETE, INSERT, UPDATE]} [WITH APPEND] [NOT FOR ENCRYPTION] AS <SQL语句组>
```

- SQL语句组一般使用`BEGIN`和`END`括起来，在语句组之前，可以是使用`IF`、`ELSE`、`WHEN`等条件关键词来进行条件选择，判断是否执行该语句组中的语句，如`IF UPDATE(SID) BEGIN ... ROLLBACK TRANSACTION END`。如果是判断错误的事务，通常在最后需要`ROLLBACK TRANSACTION`来回滚事务。
- INSTEAD OF xxx 触发器，每个表或视图仅能定义一个INSTEAD OF触发器。指定该项表示数据库不再执行触发该触发器的SQL语句，而是替换执行该触发器中的操作。它主要用户使不能更新的视图支持更新（在语句组中直接更新基本表），并且允许选择性地拒绝批处理中某些部分的操作。
- INSTEAD OF xxx 触发器不能定义在WITH CHECK OPTION的可更新视图上。同时，在含有DELETE或UPDATE级联操作定义的外键的表上也不能定义INSTEAD OF DELETE和INSTEAD OF UPDATE触发器。