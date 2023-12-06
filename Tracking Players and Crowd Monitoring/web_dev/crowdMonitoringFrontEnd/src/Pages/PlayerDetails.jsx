import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useParams } from 'react-router-dom';
import BackButton from '../components/BackButton';
import Spinner from '../components/Spinner';

const PlayerDetails = () => {
    const [player, setPlayer] = useState({});
    const [loading, setLoading] = useState(false);
    const { id } = useParams();

    useEffect(() => {
        setLoading(true);
        axios
            .get(`http://localhost:5004/players/${id}`)
            .then((res) => {
                setPlayer(res.data);
                setLoading(false);
            })
            .catch((err) => {
                console.log(err);
                setLoading(false);
            });
    }, []);

    return (
        <div className='p-4'>
            <BackButton /> 
            <h1 className='text-3xl my-4'>Player Details</h1>
            {loading ? (
                <Spinner />
            ) : (
                <div className='flex flex-col border border-sky-400 rounded-lg p-4'>
                    <div className='m-4'>
                        <span className='text-xl mr-4 text-gray-500'> ID </span>
                        <span>{player._id}</span>
                    </div>
                    <div className='m-4'>
                        <span className='text-xl mr-4 text-gray-500'> First Name </span>
                        <span>{player.firstName}</span>
                    </div>
                    <div className='m-4'>
                        <span className='text-xl mr-4 text-gray-500'> Last Name </span>
                        <span>{player.lastName}</span>
                    </div>
                    <div className='m-4'>
                        <span className='text-xl mr-4 text-gray-500'> Age </span>
                        <span>{player.age}</span>
                    </div>
                    <div className='m-4'>
                        <span className='text-xl mr-4 text-gray-500'> Team </span>
                        <span>{player.team}</span>
                    </div>
                    <div className='m-4'>
                        <span className='text-xl mr-4 text-gray-500'> Position </span>
                        <span>{player.position}</span>
                    </div>
                </div>
            )}
        </div>
    );
};

export default PlayerDetails;
