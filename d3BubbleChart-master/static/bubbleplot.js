
  categorical = [];
  numerical_barrels = [];
  numerical_highway = [];
  numerical_displ = [];

  dataSet = [];

  //take first 100 data points
  for (var i = 0; i < 50; i++) {
    categorical[i] = data[i].fields.make;
    numerical_barrels = data[i].fields.barrels08;
    numerical_highway = data[i].fields.highway08;
    numerical_displ = data[i].fields.displ;

    dataSet.push([categorical, numerical_displ, numerical_highway,  numerical_barrels]);

  }

  //get different models for coloring
  var models = getDifferentModels(categorical);

  //assign colors to models
  var dict = assignColor(models);


  //Width and height
  var width = 700;
  var height = 400;
  var padding = 30;

  //Create SVG element
  var svg = d3.select("#dataContainer")
              .append("svg")
              .attr("width", width)
              .attr("height", height);

  // text label for the x axis
  svg.append("text")
        .attr("x", 265 )
        .attr("y",  410 )
        .style("text-anchor", "middle")
        .text("Engine displacement in Liter");

  //text label for y axis
  svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 0 - 50)
        .attr("x", 0 - height/2)
        .attr("dy", "3em")
        .style("text-anchor", "middle")
        .text("Highway miles per Gallon");


  var div = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0);

  //xScale
  var xScale = d3.scaleLinear()
                .domain([1, d3.max(dataSet, function(d) { return d[1]; })])
                .range([padding, width-padding*2]);
  //yScale
  var yScale =  d3.scaleLinear()
                .domain([8,d3.max(dataSet, function(d){return d[2]; })+5])
                .range([height-padding, padding]);

  //zScale
  var zScale = d3.scaleLinear()
              .domain([0,d3.max(dataSet, function(d) { return d[3]; })])
              .range([1,30]);

  var xAxis = d3.axisBottom()
                  .scale(xScale);
  var yAxis = d3.axisLeft()
                  .scale(yScale);

  var gAxisX = svg.append('g')
                  .attr("class", "axis")
                  .attr("transform", "translate(0," + (height - padding) + ")")
                  .call(xAxis);

  var gAxisY = svg.append('g')
                  .attr("class", "yxis")
                  .attr("transform", "translate(" + padding + ",0)")
                  .call(yAxis);

  //create circles according to dataSet
  var circles = svg.selectAll("circle")
              .data(dataSet)
              .enter()
              .append("circle")
              .attr("cx", function(dataValue) {
                return xScale(dataValue[1]);
              })
              .attr("cy", function(dataValue) {
                return yScale(dataValue[2]);
              })
              .attr("r",function(dataValue) {
                return zScale(dataValue[3]);
              })
              .attr("fill", function(dataValue, i) {
                return dict[dataValue[0][i]];
              })
              .attr("fill-opacity", 0.5)
              .on("mouseover", function(dataValue, i) {
                  div.transition()
                  .duration(200)
                  .style("opacity", 0.9);
                  div.html(dataValue[0][i] + "<br/>" + Math.round(dataValue[3]))
                  .style("left", (d3.event.pageX) + "px")
                  .style("top", (d3.event.pageY - 28) + "px");
                });

  //trying animation
  circles.on("click", function() {
      var newData = createList();
      circles.data(newData)
              .transition()
              .duration(1000)
              .attr("cx", function(d, i) {return xScale((i/8)+ 1);})
              .attr("cy", function(d, i) {return yScale(d+8);});

  });

function createList() {
  var numbers = [];
  for (var i = 0; i < 50; i++) {
    numbers.push(getRandomIntInclusive(5,20));
  }

  return numbers;
}

function getRandomIntInclusive(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
}


//get different models in dataSet
function getDifferentModels(dataArray) {

  diffModels = [];

  for (var i = 0; i < dataArray.length; i++) {
    if (diffModels.indexOf(dataArray[i]) > -1) {
      continue;
    } else {
      diffModels.push(dataArray[i]);
    }
  }

  return diffModels;

}

//assigns random colors to models
function assignColor(dataArray) {

  dict = {};

  for (var i = 0; i < dataArray.length; i++) {
    dict[dataArray[i]] = getRandomColor();
  }

  return dict;
}


function getRandomColor() {
    var letters = '0123456789ABCDEF';
    var color = '#';
    for (var i = 0; i < 6; i++ ) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
}
