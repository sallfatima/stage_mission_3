// D3 Word Cloud Implementation by Eric Coopey:
// http://bl.ocks.org/ericcoopey/6382449

var source = new EventSource('/stream');
var hash = {};
var width = 1200;
var height = 700;

//update hash (associative array) with incoming word and count
source.onmessage = function (event) {
  word = event.data.split("|")[0];
  count = event.data.split("|")[1];

  if(!skip(word)){
    hash[word]=count;
  }
};

var updateViz =  function(){

bubble.value(function(d, i) { return d.value; });
colors = d3.scale.category20c(); //color category= d3.scale.category10();

  //print console message
console.log("cloudArray-1" + JSON.stringify(d3.entries(hash)));//added bu Udacity
var frequency_list={"children": d3.entries(hash)};
  //console.log(topFrequency);
  //var frequency_list ={"name":"frequencyTweets", children: [{"key":"#FASTandFURIOUS","value":250},{"key":"#IoT","value":500},{"key":"#Interracial","value":400},{"key":"#EdgeComputing","value":300},{"key":"#Stop","value":400},{"key":"#INDIANA","value":400},{"key":"#IoT6","value":100},{"key":"#LTE","value":200},{"key":"#Cuckold","value":200},{"key":"#Trump","value":200}]};

console.log(frequency_list);

              // generate data with calculated layout values
var nodes = bubble.nodes(frequency_list)
            	.filter(function(d) { return !d.children; }) // filter out the outer bubble
              .filter(function(d) { return d.value!= 0});
// Create node

var node = svg.selectAll(".node")
              .data(nodes, function(d, i) {return d.key;});

var nodeEnter = node.enter().append("g")
                .attr("class", "node")
                .attr("transform", function(d) {return "translate(" + d.x + "," + d.y + ")";});

    //Add the Circles
var circles = nodeEnter.append("circle")
            .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
            .attr("r", function(d) {return d.r;})
            .style("fill", function(d) {return colors(d.key);});


var duration = 300;
var delay = 0;

// update - this is created before enter.append. it only applies to updating nodes.
nodeEnter.append("title")
         .attr('transform', function(d, i) { return 'translate(' + d.x + ',' + d.y + ')'; })
         .text(function(d, i) { return d.key + "\n" + format(d.value); })
         .style('opacity', 0)
         .transition()
         .duration(duration * 1.2)
         .style('opacity', 1);

         //Add the Texts
nodeEnter.append("text")
         .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
         .style("text-anchor", "middle")
         .text(function(d, i) {return d.key.substring(0, d.r / 3) + ": " + d.value;})
         .style({
               "fill":"black",
               "font-family":"Helvetica Neue, Helvetica, Arial, san-serif",
               "font-size": "11px"
             });

node.select("circle")
    .transition().duration(1000)
    .attr("r", function(d) {return d.r;})
    .style("fill", function(d) {return colors(d.key);});

node.transition()
    .duration(duration)
    .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });

node.select(".value")
    .text(function(d) {return format(d.value);});

node.select("title")
    .text(function(d) {
      return d.key + " : " + format(d.value);
    });
node.select("text")
    .text(function(d, i) {return d.key.substring(0, d.r / 3) + ": " + d.value;})

node.exit().remove();


  var frequency_list1 = d3.entries(hash);

  d3.layout.cloud().size([800, 300])
  .words(frequency_list1)
  .rotate(0)
  .fontSize(function(d) { return d.value; })
  .on("end", draw)
.start();

};

// run updateViz at #7000 milliseconds, or 7 second
window.setInterval(updateViz, 7000);

//clean list, can be added to word skipping bolt
var skipList = ["https","follow","1","2","please","following","followers","fucking","RT","the","at","a","slut",""];

var skip = function(tWord){
  for(var i=0; i<skipList.length; i++){
    if(tWord === skipList[i]){
      return true;
    }
  }
  return false;
};
