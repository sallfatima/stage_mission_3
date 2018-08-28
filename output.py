
# coding: utf-8

# In[175]:


import pandas as pd
df = pd.read_excel("C:/Users/fatsall/Documents/Stage/Renault/LDA/Resultats/Renault_result_Pro_EN.xlsx",sheetname="desc_topic")
df.head()


# In[176]:


data=df.groupby(['Theme'])['topic'].sum()


# In[177]:


data=pd.DataFrame(data)


# In[178]:


dict_theme_word_cloud={}


# In[179]:


list_mot=[]797
df1=df[df['Theme']=='Divers']
for i in range(50):
    list_mot += df1[i].tolist()


# In[180]:


for theme in df.Theme:
    list_mot=[]
    df1=df[df['Theme']==theme]
    for i in range(50):
        list_mot += df1[i].tolist()
    dict_theme_word_cloud[theme]=list_mot


# In[181]:


# Libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from os import path
from PIL import Image
import numpy as np

d = "C:/Users/fatsall/Documents/Stage/EDF/Resultats"
edf_mask = np.array(Image.open(path.join(d, "edf_mask.png")))

for key,value in dict_theme_word_cloud.items():
    print("Wordcloud:{}".format(key))
    # Create a list of word
    text=' '.join(value)
    # Create the wordcloud object
    #wordcloud = WordCloud(width=480, height=480, margin=0).generate(text)
    
    fig = plt.figure(figsize=(16,12))
    # Display the generated image:
    #plt.imshow(wordcloud, interpolation='bilinear')
   # plt.axis("off")
    #plt.margins(x=0, y=0)
   # plt.show()
    
    wc = WordCloud(background_color="white", mask=edf_mask, )

    # generate word cloud
    wc.generate(text)

    # store to file
    wc.to_file(path.join(d, "images/{}.png".format(key)))

    # show
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.figure()
   # plt.imshow(edf_mask, cmap=plt.cm.gray, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# In[243]:


import webbrowser

f = open('Resultats/bubble_char/bubble_char.html','w')

message = """
<!DOCTYPE html>
<html lang="en">
<head>
		<title>Nuage des mots</title>
		<script type="text/javascript" src="https://d3js.org/d3.v4.min.js"></script>

		<link rel = "stylesheet" type = "text/css"  />



		</style>
</head>
<body>

	<script type="text/javascript">

"""
f.write(message)

f.write('dataset1 = { \n "country": "",\n')
f.write('"children": [')
for i in  range(len(data)-1):
    f.write('{"Currency":"'+str(data.index[i])+' '+str(data['topic'][i])+'", "value":' +str(data['topic'][i])+', "Number":"' +str(data['topic'][i])+ '"}, \n ')
f.write('{"Currency":"'+str(data.index[i+1])+' '+str(data['topic'][i])+'", "value":' +str(data['topic'][i+1])+', "Number":"' +str(data['topic'][i+1])+ '"} \n ')
f.write(']\n};')

message ="""
        // set color scheme and bubble chart size
        var diameter = 1200;
        var width = 1200;
        var height = 950;
        
        var color = d3.scaleOrdinal(d3.schemeCategory20);

        // set bubble parameters
        var bubble = d3.pack(dataset1)
            .size([width, height]) // should be equal to svg w and h
            .padding(3);

        var tooltip = d3.select("body")
                        .append("div")
                        .style("position", "absolute")
                        .style("z-index", "10")
                        .style("visibility", "hidden")
                        .style("color", "white")
                        .style("padding", "8px")
                        .style("background-color", "rgba(0, 0, 0, 0.75)")
                        .style("border-radius", "6px")
                        .style("font", "12px sans-serif")
                        .text("tooltip");

        // set svg parameters
        var svg = d3.select("body")
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("class", "bubble");

        svg.append("text")
            .attr("font-weight","bold")
            .attr("fill", "white")
            .attr("x", (width / 2))             
            .attr("y", 30)
            .attr("text-anchor", "middle")  
            .style("font-size", "25px") 
            .style("text-decoration", "underline")  
            .text("Nuage des topics.");

        // set nodes size correlated with dataset.cap
        var nodes = d3.hierarchy(dataset1)
            .sum(function(d) { return d.value; });

        // append node on svg, bind the data from children class
        var node = svg.selectAll(".node")
            .data(bubble(nodes).descendants())
            .enter()
            .filter(function(d){return  !d.children })
            .append("g")
            .attr("class", "node")
            .attr("transform", function(d) {
                return "translate(" + d.x + "," + d.y + ")";
            });

        node.append("circle")
            .attr("r", function(d) { return d.r; })
            .style("fill", function(d,i) { return color(i); })
            .on("click", function(d) {
                var name = d.data.Currency;
                return name; }) // pass a variable and its value to line chart.
            .on("mouseover", function(d) {
              tooltip.text(d.data.Currency + ": " + d.data.Number);
              tooltip.style("visibility", "visible");
            })
            .on("mousemove", function() { 
              return tooltip.style("top", (d3.event.pageY-10)+"px").style("left",(d3.event.pageX+10)+"px");
            })
            .on("mouseout", function(){return tooltip.style("visibility", "hidden");
            });


       node.append("text")
            .attr("x", 0)             
            
            .attr("dy", ".35em")
            .style("text-anchor", "middle")
            .text(function(d) {return d.data.Currency;})

            .call(wrap, 40)
            .attr("font-family", "Arial")
            .attr("font-size", function(d){
                return 15;
            })
            .attr("font-weight","bold")
            .attr("fill", "black");
            
            
        d3.select(self.frameElement)
            .style("height", diameter + "px");
            
        function wrap(text, width) {
            text.each(function() {
                var text = d3.select(this),
                words = text.text().split(/\s+/).reverse(),
                word,
                line = [],
                lineNumber = 0,
                y = text.attr("y"),
                dy = parseFloat(text.attr("dy")),
                lineHeight = 1.1, // ems
                tspan = text.text(null).append("tspan").attr("x", function(d) { return d.children || d._children ? -10 : 10; }).attr("y", y).attr("dy", dy + "em");     
                while (word = words.pop()) {
                    line.push(word);
                    tspan.text(line.join(" "));
                    var textWidth = tspan.node().getComputedTextLength();
                    if (tspan.node().getComputedTextLength() > width) {
                        line.pop();
                        tspan.text(line.join(" "));
                        line = [word];
                        ++lineNumber;
                        tspan = text.append("tspan").attr("x", function(d) { return d.children || d._children ? -10 : 10; }).attr("y", 0).attr("dy", lineNumber * lineHeight + dy + "em").text(word);
                    }
                }
            });
        }
        // When the user click map, the bubble updates according to the country clicked.

        var buttons = d3.select("body")
                        .append("div")
                        .attr("clasee", "countries-button")
                        .selectAll("div")
                        .data(dataset1.country)
                        .enter()
                        .append("button")
                        .text(function(d){
                            return d;
                        });
                        
                        

            buttons.on("click", function(d){
                    d3.select(node)
                        .transition()
                        .duration(500)
                        .style("background", "lightBlue");

                    update(d);
                    })

        

	</script>
</body>
</html>
"""

f.write(message)
f.close()

webbrowser.open_new_tab('Resultats/bubble_char/bubble_char.html')


# In[244]:


get_ipython().system('pip install imgkit')


# In[249]:


74615.25/60/60

