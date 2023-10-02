import React, { useState, useEffect } from 'react';
import BackButton from '../components/BackButton';
import Spinner from '../components/Spinner';
import axios from 'axios';
import { useNavigate, useParams } from 'react-router-dom';
import { useSnackbar } from 'notistack';

const UpdatePlayerStats = () => {
  const [gamesPlayed, setGamesPlayed] = useState('');
  const [goalsScored, setGoals] = useState('');
  const [assists, setAssists] = useState('');
  const [yellowCards, setYellowCards] = useState('');
  const [redCards, setRedCards] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const {id} = useParams();
  const { enqueueSnackbar } = useSnackbar();


  useEffect(() => {
    setLoading(true);
    axios.get(`http://localhost:5004/players/${id}/statistics`)
    .then((res) => {
      setGamesPlayed(res.data.gamesPlayed);
      setGoals(res.data.goalsScored);
      setAssists(res.data.assists);
      setYellowCards(res.data.yellowCards);
      setRedCards(res.data.redCards);
        setLoading(false);
      }).catch((e) => {
        setLoading(false);
        alert('An e happened. Please Chack console');
        console.log(e);
      });
  }, [])
  
  const handleUpdatePlayerStats = () => {
    const data = {
      gamesPlayed,
      goalsScored,
      assists,
      yellowCards,
      redCards,
    };
    setLoading(true);
    axios
      .put(`http://localhost:5004/players/${id}/statistics`, data)
      .then(() => {
        setLoading(false);
        enqueueSnackbar('Player Data Edited successfully', { variant: 'success' });
        navigate('/players');
      })
      .catch((e) => {
        setLoading(false);
        enqueueSnackbar('Error', { variant: 'e' });
        console.log(e);
      });
  };

  return (
    <div className='p-4'>
      <BackButton />
      <h1 className='text-3xl my-4'>Edit Player Stats</h1>
      {loading ? <Spinner /> : null}
      <div className='flex flex-col border-2 border-sky-400 rounded-xl max-w-md mx-auto p-4'>
        <div className='my-4'>
          <label className='text-xl mr-4 text-gray-500'>Games Played</label>
          <input
            type='number'
            value={gamesPlayed}
            onChange={(e) => setGamesPlayed(e.target.value)}
            className='input-field border border-gray-950'
          />
        </div>
        <div className='my-4'>
          <label className='text-xl mr-4 text-gray-500'>Goals Scored</label>
          <input
            type='number'
            value={goalsScored}
            onChange={(e) => setGoalsScored(e.target.value)}
            className='input-field border border-gray-950'
          />
        </div>
        <div className='my-4'>
          <label className='text-xl mr-4 text-gray-500'>Assists</label>
          <input
            type='number'
            value={assists}
            onChange={(e) => setAssists(e.target.value)}
            className='input-field border border-gray-950'
          />
        </div>
        <div className='my-4'>
          <label className='text-xl mr-4 text-gray-500'>Yellow Card(s)</label>
          <input
            type='number'
            value={yellowCards}
            onChange={(e) => setYellowCards(e.target.value)}
            className='input-field border border-gray-950'
          />
        </div>
        <div className='my-4'>
          <label className='text-xl mr-4 text-gray-500'>Red Card(s)</label>
          <input
            type='text'
            value={redCards}
            onChange={(e) => setRedCards(e.target.value)}
            className='input-field  border border-gray-950'
          />
        </div>
        <button
  className='button'
  onClick={handleUpdatePlayerStats}
  style={{
    backgroundColor: '#4CAF50', // Background color
    color: 'white',             // Text color
    padding: '10px 20px',       // Padding
    border: 'none',            // Remove the default border
    borderRadius: '4px',        // Rounded corners
    cursor: 'pointer',          // Cursor on hover
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)', // Add a subtle shadow
    transition: 'background-color 0.3s ease', // Smooth color transition
  }}
>
  Save
</button>

      </div>
    </div>
  );
};


export default UpdatePlayerStats;
