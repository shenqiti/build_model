'''
    网络爬虫,爬取PM2.5数据
    url:http://www.tianqihoubao.com/aqi/beijing-201703.html (201703可替换)
    未成功爬取网站：http://lishi.tianqi.com/beijing/201702.html (cookie???)
    本程序爬取2014年~至今PM2.5指数
    第一列：日期，第二列PM2.5指数
    注意：有的地方中间缺少天数,需要人工加进去
    2019/5/15
    By：Wsg
'''
import urllib.request
import re
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

OUPUT_PATH = './data/pm25.csv'

fout = open(OUPUT_PATH, 'w', encoding='utf-8')
DATE = [2014,2015,2016,2017,2018,2019]
for each in DATE:
    for i in range(1,13):
        pm25 = []
        SO2=[]
        NO2=[]
        date = []
        Co=[]
        O3=[]
        if i <=9:
            url = 'http://www.tianqihoubao.com/aqi/beijing-'+str(each)+'0'+str(i)+'.html'
        else:
            url = 'http://www.tianqihoubao.com/aqi/beijing-'+str(each)+ str(i) + '.html'
        headers = ('User-Agent',
                   'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.104 Safari/537.36')
        opener = urllib.request.build_opener()
        opener.addheaders = [headers]
        data = urllib.request.urlopen(url).read().decode('utf-8', 'ignore')
        pat = '<td>(.*?)</td>'
        result = re.compile(pat).findall(data)
        result = list(result)

        for j in range(0,len(result)):
            if (j-1)%7 == 0:
                pm25.append(result[j])
            elif (j-1)%7 == 2:
                SO2.append(result[j])
            elif (j-1)%7 ==3:
                NO2.append(result[j])
            elif (j-1)%7 ==4:
                Co.append(result[j])
            elif (j-1)%7 ==5:
                O3.append(result[j])
        for t in range (0,len(pm25)):
            if i <=9:
                if t<9:
                    date.append(str(each)+"0"+str(i)+"0"+str(t+1))
                else:
                    date.append(str(each)+"0"+str(i)+str(t+1))
            else:
                if t<9:
                    date.append(str(each)+str(i)+'0'+str(t+1))
                else:
                    date.append(str(each)+str(i)+str(t+1))

        for k in range(0,len(pm25)):
            fout.write(date[k])
            fout.write(',')
            fout.write(pm25[k])
            fout.write(',')
            fout.write(SO2[k])
            fout.write(',')
            fout.write(NO2[k])
            fout.write(',')
            fout.write(Co[k])
            fout.write(',')
            fout.write(O3[k])
            fout.write('\n')


        print("第"+str(each)+"年"+str(i)+"月"+"爬取成功")

fout.close()
