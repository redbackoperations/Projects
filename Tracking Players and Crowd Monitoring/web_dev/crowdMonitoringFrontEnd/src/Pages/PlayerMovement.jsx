import React, { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import { useParams } from 'react-router-dom';
import BackButton from '../components/BackButton';
import Spinner from '../components/Spinner';
import * as d3 from 'd3';
import L from 'leaflet';

const PlayerMovement = () => {
    const [playerMovements, setPlayerMovements] = useState([]);
    const [loading, setLoading] = useState(false);
    const { id } = useParams();
    const svgRef = useRef(null);

    useEffect(() => {
        setLoading(true);
        axios
            .get(`http://localhost:5004/players/${id}/movements`)
            .then((res) => {
                setPlayerMovements(res.data.movements);
                setLoading(false);
            })
            .catch((err) => {
                console.log(err);
                setLoading(false);
            });
    }, [id]);

    useEffect(() => {
      if (playerMovements.length > 0) {
          createLineChart();
          createBarChart();
          createMap();
      }
  }, [playerMovements]);

  const createLineChart = () => {
    const margin = { top: 20, right: 30, bottom: 30, left: 40 };
    const width = 400 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);

    svg.selectAll('*').remove(); // Clear the existing SVG content

    const parseDate = d3.timeParse('%Y-%m-%d %H:%M:%S'); // Format may vary, adjust as needed

    const x = d3.scaleTime()
        .domain(d3.extent(playerMovements, (d) => parseDate(d.date + ' ' + d.time)))
        .range([0, width]);

    const y = d3.scaleLinear()
        .domain([0, d3.max(playerMovements, (d) => d.alt)])
        .nice()
        .range([height, 0]);

    const line = d3.line()
        .x((d) => x(parseDate(d.date + ' ' + d.time)))
        .y((d) => y(d.alt));

    svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`)
        .call(d3.axisLeft(y));

    svg.append('g')
        .attr('transform', `translate(${margin.left}, ${height + margin.top})`)
        .call(d3.axisBottom(x));

    svg.append('path')
        .datum(playerMovements)
        .attr('fill', 'none')
        .attr('stroke', 'blue')
        .attr('stroke-width', 2)
        .attr('d', line);
};

const createBarChart = () => {
    const margin = { top: 20, right: 30, bottom: 30, left: 40 };
    const width = 400 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);

    const x = d3.scaleBand()
        .domain(playerMovements.map((d) => d.date))
        .range([margin.left, width + margin.left])
        .padding(0.1);

    const y = d3.scaleLinear()
        .domain([0, d3.max(playerMovements, (d) => d.sat)])
        .nice()
        .range([height, margin.top]);

    svg.selectAll('*').remove(); // Clear the existing SVG content

    svg.selectAll('rect')
        .data(playerMovements)
        .enter().append('rect')
        .attr('x', (d) => x(d.date))
        .attr('y', (d) => y(d.sat))
        .attr('width', x.bandwidth())
        .attr('height', (d) => height - y(d.sat))
        .attr('fill', 'steelblue');
    
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x));

    svg.append('g')
        .attr('transform', `translate(${margin.left}, 0)`)
        .call(d3.axisLeft(y));
};

const createMap = () => {
  const width = 400;
  const height = 300;

  // Create a Leaflet map
  const map = L.map(svgRef.current).setView([0, 0], 2); // Set the initial view coordinates and zoom level

  // Add a map tile layer (you can use different tile sources)
  L.tileLayer('http://tile.stamen.com/watercolor/{z}/{x}/{y}.jpg').addTo(map);

  // Plot player locations as markers on the map
  playerMovements.forEach((movement) => {
      const lat = parseFloat(movement.location.lat);
      const lon = parseFloat(movement.location.log);

      if (!isNaN(lat) && !isNaN(lon)) {
          L.marker([lat, lon]).addTo(map);
      }
  });

  // Set the map's dimensions
  map.invalidateSize();
  map.setMaxBounds(map.getBounds());
};


    return (
        <div className='p-4'>
            <BackButton />
            <h1 className='text-3xl my-4'>Player Movement Details</h1>
            {loading ? (
                <Spinner />
            ) : (
                <div>
                    <div>
                        {playerMovements.map((movement, index) => (
                            <div
                                key={index}
                                className='flex flex-col border border-sky-400 rounded-lg p-4 mb-4'
                            >
                                <div className='m-4'>
                                    <span className='text-xl mr-4 text-gray-500'> ID </span>
                                    <span>{movement._id}</span>
                                    <br></br>
                                    <span>{movement.time}</span>
                                    <br></br>
                                    <span>{movement.date}</span>
                                    <br></br>
                                    <span>{movement.location.lat}</span>
                                    <br></br>
                                    <span>{movement.location.log}</span>
                                    <br></br>
                                    <span>{movement.alt}</span>
                                    <br></br>
                                    <span>{movement.sat}</span>
                                    <br></br>
                                </div>
                            </div>
                        ))}
                    </div>
                    <div>
                        <h2>Altitude Over Time</h2>
                        <svg ref={svgRef} width="400" height="300"></svg>

                        <h2>Satellite Count Over Time</h2>
                        <svg ref={svgRef} width="400" height="300"></svg>
                    </div>
                </div>
            )}
        </div>
    );
};

export default PlayerMovement;
