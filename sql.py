By:shenqiti
2019/9/1
'''
利用Python进行数据库操作
'''

'''
连接MySQL数据库

import pymysql.cursors
#使用pymysql指令来连接数据库
connection=pymysql.connect(host='',user='',password='',db='',charset='',cursorclass=pymysql.cursors.DictCursor
)
host:要连接的数据库的IP地址
user：登录的账户名，如果登录的是最高权限账户则为root
password：对应的密码
db：要连接的数据库，如需要访问上节课存储的IRIS数据库，则输入'IRIS'
charset：设置编码格式，如utf8mb4就是一个编码格式
cursorclass：返回到Python的结果，以什么方式存储，如Dict.Cursor是以字典的方式存储

创建新的数据

try:
    #从数据库链接中得到cursor的数据结构
    with connection.cursor() as cursor:
    #在之前建立的user表格基础上，插入新数据，这里使用了一个预编译的小技巧，避免每次都要重复写sql的语句
        sql="INSERT INTO `USERS`(`email`,`password`) VALUES (%s,%s)"
        cursor.execute(sql,('webmaster@python.org','very_secret'))
    #执行到这一行指令时才是真正改变了数据库，之前只是缓存在内存中

    connection.commit()


调用数据：查询webmaster@python.org邮箱的密码

    with connection.cursor() as cursor:
        sql = "SELECT `id`,`password` FROM `user` WHERE `email`=%s"
        cursor.execute(sql,('webmaster@python.org',))
        #只取出一条结果
        result=cursor.fetchone()
        print(result)

#最后别忘了关闭连接
finally:
    connection.close()

最后调用的结果为

{'password': 'very-secret', 'id': 1} 
COMMIT: 
1.注意过于频繁的commit会降低数据插入的效率，可以在多行insert之后一次性commit；2.autocommit选项：默认每一个insert操作都会触发commit操作方式，是在pymysql.connect的db参数后面，加一个autocommit=True参数
'''

import pymysql.cursors
connection=pymysql.connect(host='___',user='___',password='___',db='iris',charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor)
try:
    with connection.cursor() as cursor:
        sql= " SELECT * FROM `iris_with_id` WHERE `id`=%s"
        cursor.execute(sql,('3',))
        result=cursor.fetchone()
        print(result)
        print(result['id'])
finally:
    connection.close()
    
    
import pymysql.cursors
connection=pymysql.connect(host='___',user='___',password='___',db='iris',charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor)
try:
    with connection.cursor() as cursor:
        SELECT * FROM `iris_with_id` WHERE `petal_width`>0.5
        result=cursor.fetchall()
        print(result)
finally:
    connection.close()
    
import pymysql.cursors
connection=pymysql.connect(host='___',user='___',password='___',db='iris',charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor)
try:
    with connection.cursor() as cursor:
        SELECT * FROM  `iris_with_id` WHERE `petal_width` >0.5
        result=cursor.fetchall()
        print(len(result))
        for each_r in result:
            print(each_r['id'])
finally:
    connection.close()
    
    
