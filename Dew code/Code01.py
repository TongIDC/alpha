from pymodbus.client.sync import ModbusTcpClient
import pymysql

host = '192.168.2.103'
port = '502'

client = ModbusTcpClient(host, port)
a = client.connect()

print(a)

rr = client.read_holding_registers(30775,2,unit=3)
assert(rr.function_code < 0x80)     # test that we are not an error
print (rr)
print (rr.registers)


lenght = 'Watt'
value = rr.registers[1]

print (lenght)
print (value)

c = pymysql.connect(host='localhost', user='root', passwd='lago26', db='Chula')
try:
    with c.cursor() as cursor:
        sql = "INSERT INTO `TEST` (`no`,`no1`) VALUES (%s,%s)"
        cursor.execute(sql, (value,lenght))

        c.commit()
finally:
    c.close()
