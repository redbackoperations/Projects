import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useParams } from 'react-router-dom';
import BackButton from '../components/BackButton';
import Spinner from '../components/Spinner';
import Plot from 'react-plotly.js';

const PlayerMovement = () => {
    const [playerMovements, setPlayerMovements] = useState([]);
    const [loading, setLoading] = useState(false);
    const { id } = useParams();

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

    // Data for the line chart
    const lineChart = {
        x: playerMovements.map((movement) => movement.date),
        y: playerMovements.map((movement) => movement.alt),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Altitude',
    };

    // Data for the bar chart
    const barChart = {
        x: playerMovements.map((movement) => movement.date),
        y: playerMovements.map((movement) => movement.sat),
        type: 'bar',
        name: 'Satellite Count',
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
                        <Plot
                            data={[lineChart]}
                            layout={{
                                width: 400,
                                height: 300,
                                title: 'Altitude vs. Date',
                                xaxis: { title: 'Date' },
                                yaxis: { title: 'Altitude' },
                            }}
                        />

                        <h2>Satellite Count Over Time</h2>
                        <Plot
                            data={[barChart]}
                            layout={{
                                width: 400,
                                height: 300,
                                title: 'Satellite Count vs. Date',
                                xaxis: { title: 'Date' },
                                yaxis: { title: 'Satellite Count' },
                            }}
                        />
                    </div>
                </div>
            )}
        </div>
    );
};

export default PlayerMovement;
