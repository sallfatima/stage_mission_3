#!/usr/bin/python

import matplotlib.pyplot as plt

system_name = list()
law_level = list()
accept_level = list()
starport = list()
tech_level = list()

hex_to_num = {  '0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,
                '8':8,'9':9,'A':10,'B':11,'C':12,'D':13,'E':14,'F':15,'G':16}

tech_level_rating = {   '0':10,'1':10,'2':10,'3':10,'4':10,
                        '5':100,'6':100,'7':100,'8':100,'9':100,
                        'A':1000,'B':1000,'C':1000,'D':1000,'E':1000,
                        'F':1000,'G':1000,'H':1000}
                        
starport_to_color = {'A':10,'B':9,'C':8,'D':7,'E':6,'X':5}
                        



i = 0

for line in open("bubblesec.txt"):
    i += 1
    systems_ = str(line)
    if i == 1:
        name_char = systems_.find('Name')
        law_char = systems_.find('UWP') + 6
        tech_char = systems_.find('UWP') + 8
        accept_char = systems_.find('[') + 2
        starport_char = systems_.find('UWP')
    if i > 2:

        system_name.append(systems_[name_char:name_char+5])
        law_level.append(hex_to_num[systems_[law_char]])
        accept_level.append(hex_to_num[systems_[accept_char]])
        tech_level.append(tech_level_rating[systems_[tech_char]])
        starport.append(starport_to_color[systems_[starport_char]])




plt.xlabel('Law Level')
plt.ylabel('Acceptance Level')
plt.title('Acceptance and Law')

plt.axis([-1, 17, -1, 17])
plt.scatter(law_level,accept_level,s=tech_level, c = starport, cmap=plt.cm.RdYlGn)

for i, txt in enumerate(system_name):
	plt.annotate(txt, (law_level[i]-.5,accept_level[i]))
	
plt.show()