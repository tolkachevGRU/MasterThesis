import mysql.connector
from joblib import Parallel, delayed

def getConnection():
    return mysql.connector.connect(user='root', password='g33nwifiindeclub',
								  host='localhost',port='3306',
								  database='dataset',charset="utf8", use_unicode = True)

def query_mysql(query, headers):
	cnx = getConnection()
	cursor = cnx.cursor()
	cursor.execute(query)
	#get header and rows
	header = [i[0] for i in cursor.description]
	rows = [list(i) for i in cursor.fetchall()]
	#append header to rows
	if (headers):
		rows.insert(0,header)
	cursor.close()
	cnx.close()
	return rows

def query_headers(query):
	cnx = getConnection()
	cursor = cnx.cursor()
	cursor.execute(query)
	#get header and rows
	header = [i[0] for i in cursor.description]
	rows = [list(i) for i in cursor.fetchall()]
	cursor.close()
	cnx.close()
	return header

def addRow(row):
	newrow = u'<tr>' 
	newrow += u'<td align="left" style="padding:1px 4px">'+str(row[0])+u'</td>'
	row.remove(row[0])
	newrow = newrow + ''.join([u'<td align="right" style="padding:1px 4px">' + str(x) + u'</td>' for x in row])  
	newrow += '</tr>\n' 
	return newrow

#take list of lists as argument	
def nlist_to_html(list2d, file):
	textfile = open(file, "w")
	#bold header
	textfile.write('<table border="1" bordercolor=000000 cellspacing="0" cellpadding="1" style="table-layout:fixed;vertical-align:bottom;font-size:13px;font-family:verdana,sans,sans-serif;border-collapse:collapse;border:1px solid rgb(130,130,130)" >')
	# list2d[0] = [u'<b>' + i + u'</b>' for i in list2d[0]] 
	for tablerow in Parallel(n_jobs=16, verbose=1)(delayed(addRow)(row) for row in list2d):
		textfile.write(tablerow)
	# for row in list2d:
		# htable += addRow(row)
	textfile.write('</table>')
	
 
def sql_html(query, headers, file):
	nlist_to_html(query_mysql(query, headers), file)