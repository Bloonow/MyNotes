> 值得注意的是，本笔记中的一些SQL语句在很大程度上和《SQL Server》里的一样，实际上它们都遵守一定的SQL规范，这些SQL语句在一定程度上是通用的。

[toc]

# MySQL 数据类型

MySQL对数据字段类型的支持可大致可以分为三类：数值、日期/时间、和字符串（字符）类型。

## （一）数值类型

MySQL支持所有标准SQL数值数据类型。

这些类型包括严格数值数据类型（INTEGER、SMALLINT、DECIMAL、和NUMERIC），以及近似数值数据类型（FLOAT、REAL、和DOUBLE PRECISION），其中关键字INT是INTEGER的同义词，关键字DEC是DECIMAL的同义词。

BIT数据类型保存位字段值，并且支持MyISAM、MEMORY、InnoDB、和BDB表。

作为SQL标准的扩展，MySQL也支持整数类型TINYINT、MEDIUMINT、和BIGINT。

下面的表显示了需要的每个整数类型的存储和范围。

|     类型     |                   大小                   |                        范围（有符号）                        |                        范围（无符号）                        |      用途       |
| :----------: | :--------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-------------: |
|   TINYINT    |                  1 byte                  |                         (-128，127)                          |                           (0，255)                           |    小整数值     |
|   SMALLINT   |                 2 bytes                  |                      (-32 768，32 767)                       |                         (0，65 535)                          |    大整数值     |
|  MEDIUMINT   |                 3 bytes                  |                   (-8 388 608，8 388 607)                    |                       (0，16 777 215)                        |    大整数值     |
| INT或INTEGER |                 4 bytes                  |               (-2 147 483 648，2 147 483 647)                |                      (0，4 294 967 295)                      |    大整数值     |
|    BIGINT    |                 8 bytes                  |   (-9 223 372 036 854 775 808，9 223 372 036 854 775 807)    |               (0，18 446 744 073 709 551 615)                |   极大整数值    |
|    FLOAT     |                 4 bytes                  | (-3.402 823 466 E+38，-1.175 494 351 E-38)，0，(1.175 494 351 E-38，3.402 823 466 351 E+38) |         0，(1.175 494 351 E-38，3.402 823 466 E+38)          | 单精度 浮点数值 |
|    DOUBLE    |                 8 bytes                  | (-1.797 693 134 862 315 7 E+308，-2.225 073 858 507 201 4 E-308)，0，(2.225 073 858 507 201 4 E-308，1.797 693 134 862 315 7 E+308) | 0，(2.225 073 858 507 201 4 E-308，1.797 693 134 862 315 7 E+308) | 双精度 浮点数值 |
|   DECIMAL    | 对DECIMAL(M,D) ，如果M>D，为M+2否则为D+2 |                        依赖于M和D的值                        |                        依赖于M和D的值                        |     小数值      |

## （二）日期和时间类型

表示时间值的日期和时间类型为DATETIME、DATE、TIMESTAMP、TIME、和YEAR。TIMESTAMP类型有专有的自动更新特性，将在后面描述。

每个时间类型有一个有效值范围和一个"零"值，当指定不合法的MySQL不能表示的值时使用"零"值。

|   类型    | 大小 ( bytes) |                             范围                             |        格式         |           用途           |
| :-------: | :-----------: | :----------------------------------------------------------: | :-----------------: | :----------------------: |
|   DATE    |       3       |                   1000-01-01 ~ 9999-12-31                    |     YYYY-MM-DD      |          日期值          |
|   TIME    |       3       |                    -838:59:59 ~ 838:59:59                    |      HH:MM:SS       |     时间值或持续时间     |
|   YEAR    |       1       |                         1901 ~ 2155                          |        YYYY         |          年份值          |
| DATETIME  |       8       |          1000-01-01 00:00:00 ~ 9999-12-31 23:59:59           | YYYY-MM-DD HH:MM:SS |     混合日期和时间值     |
| TIMESTAMP |       4       | 1970-01-01 00:00:00 ~ 2038结束时间是第2147483647秒，北京时间2038-1-19 11:14:07，格林尼治时间 2038年1月19日 凌晨 03:14:07 |   YYYYMMDD HHMMSS   | 混合日期和时间值，时间戳 |

## （三）字符串类型

|    类型    |          大小           |              用途               |
| :--------: | :---------------------: | :-----------------------------: |
|    CHAR    |      0 ~ 255 bytes      |           定长字符串            |
|  VARCHAR   |     0 ~ 65535 bytes     |           变长字符串            |
|  TINYBLOB  |      0 ~ 255 bytes      | 不超过 255 个字符的二进制字符串 |
|  TINYTEXT  |      0 ~ 255 bytes      |          短文本字符串           |
|    BLOB    |    0 ~ 65 535 bytes     |     二进制形式的长文本数据      |
|    TEXT    |    0 ~ 65 535 bytes     |           长文本数据            |
| MEDIUMBLOB |  0 ~ 16 777 215 bytes   |  二进制形式的中等长度文本数据   |
| MEDIUMTEXT |  0 ~ 16 777 215 bytes   |        中等长度文本数据         |
|  LONGBLOB  | 0 ~ 4 294 967 295 bytes |    二进制形式的极大文本数据     |
|  LONGTEXT  | 0 ~ 4 294 967 295 bytes |          极大文本数据           |

值得注意的是，Char(n)和Varchar(n)中括号中n代表字符的个数，并不代表字节个数，比如CHAR(30)就可以存储30个字符；但它们所占存储空间的字节数是与相应的字符以及其编码有关的。

CHAR和VARCHAR类型类似，但它们保存和检索的方式不同，它们的最大长度和是否尾部空格被保留等方面也不同。在存储或检索过程中不进行大小写转换。

BINARY和VARBINARY类似于CHAR和VARCHAR，不同的是它们包含二进制字符串而不是非二进制字符串。故它们没有字符集，并且排序和比较基于列值字节的数值值。

BLOB是一个二进制大对象，可以容纳可变数量的数据，有4种BLOB类型：TINYBLOB、BLOB、MEDIUMBLOB、LONGBLOB。它们区别在于可容纳存储范围不同。

有4种TEXT类型：TINYTEXT、TEXT、MEDIUMTEXT、LONGTEXT。对应的这4种类型，可存储的最大长度不同，可根据实际情况选择。

# MySQL 语法规则

需要注意的是，在Windows下，字符串等某些字段值用的是 **'** ，而字段名或者表名等使用的是 **`** ，应该注意它们之间的区别。

## （一）基本语句

### 1. 增

```mysql
create database DatabaseName;
```

- 创建数据库。

```mysql
create table TableName (ColumnName ColumnType [Options], ...);

CREATE TABLE IF NOT EXISTS 'Book' (
   `id` INT UNSIGNED AUTO_INCREMENT,
   `title` VARCHAR(100) NOT NULL,
   `author` VARCHAR(40) NOT NULL,
   `date` DATE,
   PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
```

- Options可以有多种可选值，如进一步说明数据类型、限制数值、定义键值等。
- 如果要使用无符号整数，可以使用`int unsigned`。
- `AUTO_INCREMENT`定义列为自增的属性，一般用于主键，数值会自动加1。
- 如果限制字段不能为空，Options可为`not null`。
- 在最后可以使用`PRIMARY KEY(ColumnName)`关键字定义列为主键，如果有多列，则以逗号分隔。
- `ENGINE`设置存储引擎，`CHARSET`设置编码，可选。

```mysql
insert into TableName (field1, field2, ...) values (value1, value2, ...), ...;
```

- 向一个表中插入数据，如果数据是字符型，必须使用单引号或者双引号。
- 如果某一位置的主键是AUTO_INCREMENT自增的，则相应位置上可以传入0或者null，这样数据可以自增。

### 2. 删

```mysql
drop database DatabaseName;
```

- 删除数据库。

```mysql
drop table TableName;
```

- 删除数据表。

```mysql
delete from TableName [where Clause];
```

- 在数据表中删除记录。
- 可以使用where子句指定任何条件；如果没有指定where子句，则会删除表中所有的记录。

### 3. 改

```mysql
alter table TableName engine = MYISAM;
```

- 修改数据库引擎，上述MYISAM只是个例子，修改时请务必了解清楚。

修改数据表的定义，删除外键约束。

```mysql
alter table TableName drop foreign key KeyName;
```

- 删除该数据表的某个外键KeyName。

修改数据表的定义，增加或删除字段。

```mysql
alter table TableName [drop | add] [ColumnName [Type]];
```

- 如果数据表中只剩余一个字段则无法使用drop来删除字段。
- 向表中增加字段时需要同时指定它的数据类型。
- 添加的字段默认加在最后，可以在新添加的字段后跟`first`或者`after ExistedColumn`来指定新字段的位置。

```mysql
alter table TableName modify Column1 Type1 first | after Column2;
```

- 修改字段的相对位置。

修改数据表的定义，修改已存在字段。

```mysql
alter table TableName modify existedColumn newType [default v];
alter table TableName change oldColumnName newName newType [default v];
```

- 修改某一个已存在字段的名字、类型（包括默认值）。
- 在类型后可跟一些值的约束，如not null等。

修改某个字段的默认值，设置或者删除。

```mysql
alter table TableName alter existedColumn set default v;
alter table TableName alter existedColumn drop default;
```

- 可以使用alter来修改字段的默认值，或者结合drop来删除字段的默认值。

```mysql
alter table TablaName rename to NewTableName;
```

- 修改数据表的名称。

```mysql
update TableName set field1 = newValue1, ... [where Clause];
```

- 修改或更新数据表中的数据。
- 可以同时更新一个或多个字段。
- 可以使用where子句指定任何条件；如果没有where子句，则会更新表中所有的记录。

### 4. 查

```mysql
select ColumnName, ... from TableName [where Clause] [limit f,t] [offset M];
```

- 查询数据库中数据表中的数据，可以使用一个或多个表，表之间用逗号`,`分割，并使用where子句来设定查询条件。
- 可以使用`*`来代替其他字段，select语句会返回表的所有字段数据。
- 可以使用where子句指定任何条件；如果没有where子句，则会查询表中所有的记录。
- 可以使用limit属性来设定返回的记录数；使用offset指定select语句开始查询的数据偏移量，默认情况下偏移量为0。

### 5. 其他操作

```mysql
use TableName;        # 使用/选择当前数据库
show databases;        # 列出 MySQL 数据库管理系统的数据库列表
show tables [from DatabaseName];    # 显示指定数据库的所有表
show columns from TableName;        # 显示数据表属性列的信息
show index from TableName;            # 显示数据表索引的信息
show table status [from DatabaseName] [like 'regStr'] [\G];        # 显示数据库的信息
```

## （二）扩展性语句

### 1. where子句

where子句可以用来限定某些条件，如在select、delete、update等语句中都能得到应用，它之前常为from引出的表，如下所示。

```mysql
from TableName1, ... where Condition1 [and [or]] Condition2...;
```

- 语句中可以使用一个或者多个表，表之间使用逗号`,`分割，并使用where语句来设定查询条件。
- 在where子句中指定任何条件，可用and或者or指定一个或多个条件。
- 条件中可以根据数据表的字段来使用六种比较操作符，如`<`、`<=`、`>`、`>=`、`=`、`!=`、`<>`。

### 2. like子句

like可以用在where子句中，其主要用在匹配字符串类型的字段，类似于正则表达式，如下所示。

- `%`，表示任意0个或多个字符。可匹配任意类型和长度的字符，有些情况下若是中文，可能需要使用两个百分号`%%`表示。
- `_`，表示匹配单个任意字符，它常用来限制表达式的字符长度语句。
- `[]`，表示括号内所列字符中的一个（类似正则表达式）。指定一个字符、字符串或范围，要求所匹配对象为它们中的任一个。
- `[^]`，表示不在括号所列之内的单个字符，匹配对象为指定字符以外的任一个字符。
- 查询内容包含通配符时，对特殊字符`%`、`_`、`[`等的查询语句无法正常实现，需要把特殊字符用`[]`括起以便正常查询，类似于转义字符。

### 3. union操作符

union操作符用于连接两个以上的SELECT语句的结果组合到一个结果集合中，需要这些结果保持字段之间的对应。

```mysql
select ColumnSeq from TableSeq [where Conds]
union [all | distinct]
select ColumnSeq from TableSeq [where Conds];
```

- distinct，表示删除结果集中重复的数据。默认情况下union操作符已经删除了重复数据，所以distinct修饰符对结果没什么影响。
- all，返回所有结果集，包含重复数据。

### 4. order by排序

可以使用order by子句来对读取的数据进行排序，再返回搜索结果。

```mysql
select ColumnSeq from TableSeq [where Conds]
order by Column1 [asc | desc], [Column2...];
```

- 可以使用任何（多个）字段来作为排序的条件，从而返回排序后的查询结果。
- asc和desc关键字来设置查询结果是按升序（默认）或降序排列。

### 5. group by分组

group by语句根据一个或多个列对结果集进行分组，在分组的列上可以使用count、sum、avg等函数。

```mysql
select ColumnName, function(ColumnName) from TableName where [Clause]
group by ColumnName [with rollup];
```

- with rollup可以实现在分组统计数据基础上再进行相同的统计（sum、avg、count等）。

使用分组时，可以支持使用`having`进行分组过滤，如下。

```mysql
select count(*) as aNum from aTable group by aColumn having aNum > 1;
```

### 6. join连接

join关键字主要用在多表查询，用于连接多个表，根据连接和保留元组的策略，可分为以下三种：

- `inner join`（内连接或等值连接），获取连个表中相应字段匹配的记录。可省略inner。
- `left join`（左连接），会保留左表中指定字段的所有元组，即使它在右表中没有对应匹配值。
- `right join`（右连接），会保留右表中指定字段的所有元组，即使它在左表中没有对应匹配值。

## （三）进阶操作

### 1. null值处理

使用select命令及where子句来读取表中数据的时候，若作为查询条件的字段为null时，需要额外注意，MySQL对null提供了三大运算符：

- `is null`，当列的值是null，此运算符返回true。
- `is not null`，当列的值不是null，此运算符返回true。

- `<=>`，比较操作符（不同于`=`运算符），当比较的的两个值相等或者都为null时返回true。

关于null的条件比较运算是比较特殊的。在MySQL中，null与任何其它值的比较（即使是null）永远返回null。

```mysql
ifnull(ColumnValue, aValue);
```

- 函数，当列的值为null时，把它替换成aValue。

```mysql
coalesce(valueA, valueB, ...);
```

- 返回一系列值中第一个不是null的值，如果都为null则返回null。

### 2. 正则表达式

详细的正则表达式知识可以参看《C++高级编程》。

在MySQL中除了使用like进行模糊字符串匹配外，还可以使用`regexp`操作符来进行正则表达式匹配，如下。

```mysql
... where aString regexp 'regStr';
```

- 其中regStr即正则表达式模式字符串。

### 3. 复制数据表

如果需要完全的复制MySQL的数据表，包括表的结构、索引、默认值等，仅使用create table ...命令，是无法实现的。而应该使用如下步骤。

1. 使用`show create table TableName`方法获取创建数据表的SQL语句，该语句包含了原有数据表的结构、索引等。
2. 复制上述命令返回的SQL语句，修改数据表名，并执行SQL语句，通过该SQL命令将完全复制数据表结构。
3. 若想复制表的内容，可用`insert into NewTable select * form OldTable`等之类的语句来实现。

执行以上步骤后，将完整的复制表，包括表结构及表数据，也可以通过一下两部完整复制一个数据表：

```mysql
create table NewTableName like OldTableName;
insert into NewTableName select * from OldTableName;
```

可以在创建表示指定一些其他表的字段信息或元组，如下。

```mysql
create table NewTableName as (
    select Column1 [as newColumn1], ... from SomeTable [where Conds]
);
```

或者可以在创建表的同时定义表中的字段信息，如下。

```mysql
create table NewTableName (
    NewColumnName Type [constraints], ...
) as (
    select xxx
);
```

### 4. 处理重复数据

1. **防止表中出现重复数据**

MySQL数据表中可能存在重复的记录，如果不允许某字段出现重复记录，则可以将其设置为`primary key`（主键）或者`unique`（唯一）。如果设置了唯一索引，那么在insert插入重复数据时，SQL语句将无法执行成功，并抛出错误。

而`insert ignore into`语句同样是插入数据，对于已存在的重复数据，它会忽略不进行插入；对于不存在的数据才插入。如果某字段设置了唯一索引，那么对于重复的插入，insert ignore into不会返回错误而是以警告形式返回。

而`replace into`语句，如果存在重复的指定唯一性的记录，则先删除原来的记录，再插入新的记录。

2. **统计重复数据**

一般情况下，查询重复的值，可以执行以下操作：确定哪一列包含的值可能会重复，使用count(*)列出作为选择条件，在group by子句根据列分组，使用having子句将统计列的重复数大于1。

3. **过滤重复数据**

如果需要读取不重复的数据可以在select语句中使用`distinct`关键字，用它来修饰某列名，来过滤重复数据。也可以使用group by子句来读取数据表中不重复的数据。

4. **删除重复数据**

如果想删除数据表中重复的数据，可以先过滤掉重复数据，创建一个新表，并删除原来的数据表，然后将新表重命名为原来的表。如下。

```mysql
create table tmp select * from aTable group by (ColumnList);
drop table aTable;
alter table tmp rename to aTable;
```

当然也可以通过在数据表中添加唯一索引或者主键，来删除表中重复的数据。

```mysql
alter ignore table aTable add primary key (ColumnList);
```

# MySQL 功能

## （一）索引

索引分为两种：单列索引，即一个索引只包含单个列，一个表可以有多个单列索引；组合索引，即一个索引包含多个列。

创建索引时，需要确保该索引是应用在SQL查询语句的条件（一般作为where子句的条件）。实际上，索引也是一张表，该表保存了记录的索引字段与记录的主键，并指向实体表的记录。

过多的索引会降低更新表的速度，索引文件也会占用磁盘空间。

### 1. 基本操作

普通索引是最基本的索引，它没有任何限制；唯一索引的索引列的值必须唯一，但允许有空值，如果是组合索引，则列值的组合必须唯一；主键就是不能为空的特殊唯一索引。

```mysql
create [unique] index IndexName on TableName(ColumnName, ...);
```

- 直接在表上的某列上创建索引。

```mysql
alter table TableName add [unique] index IndexName(ColumnName, ...);
```

- 修改表的结构，添加索引。

```mysql
create table TablaName (ColumnList,
    [unique] index IndexName (ColumnName), ...
);
```

- 创建表的时候直接指定索引。

```mysql
drop index IndexName on TableName;
```

- 删除某个表上的索引。

```mysql
show index from TableName; [\G]
```

- 显示表上的索引信息，\G是用来格式化输出。

### 2. 使用alter操作

有四种方式来添加数据表的索引，如下。

```mysql
alter table TableName add primary key (ColumnList);
alter table TableName drop primary key;
```

- 上一个语句添加一个主键，这意味着索引值必须是唯一的，且不能为null，主键只能作用于一个列上。
- 后一个语句用来删除表上的主键。

```mysql
alter table TableName add unique IndexName (ColumnList);
```

- 这条语句创建索引的值必须是唯一的（除了null外，null可能会出现多次）。

```mysql
alter table TableName add index IndexName (ColumnList);
```

- 添加普通索引，索引值可出现多次。

```mysql
alter table TableName add fulltext IndexName (ColumnList);
```

- 该语句指定了索引为fulltext，用于全文索引。

## （二）事务

MySQL事务主要用于处理操作量大，复杂度高的数据。只有使用了InnoDB数据库引擎的数据库或表才支持事务，事务用来管理insert、update、delete语句。

```mysql
begin | start transaction
```

- 显示地开启一个事务。

```mysql
commit [work]
```

- 提交事务，并使已对数据库进行的所有修改成为永久性的。

```mysql
rollback [work]
```

- 回滚，结束用户的事务，并撤销正在进行的所有未提交的修改。

```mysql
savepoint SavePointName
```

- 在事务中创建一个保留点，一个事务中可以有多个保留点。savepoint是在数据库事务处理中实现“子事务”（subtransaction），也称为嵌套事务的方法。

```mysql
rollback to SavePointName
```

- 把事务回滚到保留点。事务可以回滚到savepoint而不影响savepoint创建前的变化，不需要放弃整个事务。

```mysql
release savepoint SavePointName
```

- 删除一个事务的保留点，当没有指定的保留点时，执行该语句会抛出一个异常。或者保留点在事务处理完成（执行一条rollback或commit）后自动释放。

```mysql
set transaction isolation level aLevel
```

- 设置事务的隔离级别。InnoDB存储引擎提供事务的隔离级别有`read uncommitted`、`read committed`、`repeatable read`、`serialzable`。

```mysql
set autocommit = 0
set autocommit = 1
```

- 禁止/开启自动提交。

在MySQL命令行的默认设置下，事务都是自动提交的，即执行SQL语句后就会马上执行commit操作；因此要显式地开启一个事务务须使用命令begin或start transaction操作，或者执行命令set autocommit=0，用来禁止使用当前会话的自动提交。

## （三）临时表

MySQL临时表在需要保存一些临时数据时是非常有用的，他只在当前连接可见，当关闭连接时，MySQL会自动删除表并释放所有临时表空间。

- 如果你使用PHP脚本来创建MySQL临时表，当PHP脚本执行完成后，该临时表会自动销毁。
- 如果使用客户端程序连接MySQL数据库服务器来创建临时表，那么当关闭客户端程序时会销毁临时表，当然也可以手动销毁。

值得注意的是，如果使用`show tables`命令来查看数据表时，是没有临时表的。

临时表除了是临时的之外，其他都和基本表一样，区别是基本表的SQL语句用到`table`的地方，临时表的SQL语句要使用`temporary table`，例如：

```mysql
create temporary table TmpTableName(...);
create temporary table TmpTableName as (
    select xxx from xxx [Conds]
);
```

- 上面显示了两种创建临时表的方式，可以直接创建一个临时表，然后再插入数据；也可以直接根据查询语句返回的结果集来构造临时表。

```mysql
drop table TmpTableName;
```

- 可以手动删除临时表。

## （四）序列

MySQL序列是一组整数如1,2,3,...，由于一张数据表只能有一个字段自增主键，若要其他字段也实现自动增加，就可以使用MySQL序列来实现。

最简单的方法就是使用`auto_increment`来定义列，在插入自增字段时，可以指定0或者null让其自动增长。在SQL中可以通过`last_insert_id()`函数来获得最后的插入表的自增列的值。

如果删除了数据表中的多条记录，并希望对剩下数据的auto_increment列进行重新排列，那么可以通过删除自增的列，然后重新添加来实现。 如下所示：

```mysql
alter table TableName drop AIncrColumn;
alter table TableName
    add AIncrColumn int unsigned not null auto_increment first,
    add primary key (AIncrColumn);
```

- 该操作要非常小心，如果在删除的同时又有新记录添加，有可能会出现数据混乱。

一般情况下序列的开始值为1，如果需要指定一个开始值，可以在创建表时使用如下语句来实现：

```mysql
create table TableName (...) auto_increment=startVale;
```

或者在表创建成功后，通过以下语句来实现：

```mysql
alter table TableName auto_increment = startValue;
```

## （五）函数

MySQL提供了很对内置函数，如字符串函数、数字函数、如期函数、还有一些高级函数。由于内置函数过多，这里篇幅有限就不再列出，使用时可以参看相关资料，如菜鸟教程的[MySQL函数](https://www.runoob.com/mysql/mysql-functions.html)。