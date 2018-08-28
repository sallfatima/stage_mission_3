# course-visualiser
<h2>Interactive Visualisation: Bubble chart using d3.js</h2>

In this bubble chart, each course is drawn as a circle on a two or multiple dimensional plot. Each course in the database has the following features: Mode, Level, Sector, Price, Duration, Langue. These features will be represented through different dimensions in the visualisation.
      <p>
        <h3>X-axis: Mode</h3>
      The feature Mode can take one of the values: Distance, Continu, Cours du soir, Cours d’emploi, Partiellement presentiel, and forms the X-axis of the plot.
  		</p>
  		<p>
  			<h3>Y-axis: Level</h3>
			The feature Level can take one of the values: Advance, Cours specialises, Formation profession- nelle superieure, Sport-etudes, and forms the Y-axis of the plot.
  		</p>
  		<p>
  			<h3>Bubble size: Price</h3>
			The radius of the bubble is a function of the price of the course.
  		</p>
  		<p>
  			<h3>Mouseover: Course information</h3>
			The other features such as Duration, Langue, Course name will be shown when the mouse is hovered over a particular bubble. 
  		</p>
  		<p>
  			<h3>Pulsating bubbles: Trending courses</h3>
			Based on web scraping, a few courses in the database was found to be trending or popular. These courses will be shown as pulsating bubbles in the plot.
