import pymysql.cursors
import pymysql
import pymysql.cursors


config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'root',
    'db': 'test',
    'charset': 'utf8',
    'cursorclass': pymysql.cursors.DictCursor,
}

connection = pymysql.connect(**config)
connection.autocommit(True)
cursor = connection.cursor()

def get_table_list():
    results=[]
    cursor.execute('show tables from test;')
    query_result = cursor.fetchall()
    for i in query_result:
        results.append(i['Tables_in_test'])
    return results

def get_data(sets_id):
    sql='select * from kpdata where id='+sets_id+';'
    cursor.execute(sql)
    query_result = cursor.fetchall()
    return query_result