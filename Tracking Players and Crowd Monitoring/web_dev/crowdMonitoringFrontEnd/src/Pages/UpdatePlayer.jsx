import React, { useState, useEffect } from 'react';
import BackButton from '../components/BackButton';
import Spinner from '../components/Spinner';
import axios from 'axios';
import { useNavigate, useParams } from 'react-router-dom';
import { useSnackbar } from 'notistack';

const UpdatePlayer = () => {
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [age, setAge] = useState('');
  const [team, setTeam] = useState('');
  const [position, setPosition] = useState('');
  const [DOB, setDOB] = useState('');
  const [sport, setSport] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const {id} = useParams();
  const { enqueueSnackbar } = useSnackbar();


  useEffect(() => {
    setLoading(true);
    axios.get(`http://localhost:5004/players/${id}`)
    .then((res) => {
      setFirstName(res.data.firstName);
      setLastName(res.data.lastName);
      setAge(res.data.age);
      setTeam(res.data.team);
      setPosition(res.data.position);
      setDOB(res.data.dateOfBirth);
      setSport(res.data.sport);
        setLoading(false);
      }).catch((e) => {
        setLoading(false);
        alert('An e happened. Please Chack console');
        console.log(e);
      });
  }, [])
  
  const handleUpdatePlayer = () => {
    const data = {
      firstName,
      lastName,
      age,
      team,
      position,
      dateOfBirth: DOB,
      sport,
    };
    setLoading(true);
    axios
      .put(`http://localhost:5004/players/${id}`, data)
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
      <h1 className='text-3xl my-4'>Edit Player</h1>
      {loading ? <Spinner /> : ''}
      <div className='flex flex-col border-2 border-sky-400 rounded-xl w-[600px] p-4 mx-auto'>
      <div className='my-4'>
          <label className='text-xl mr-4 text-gray-500'>First Name</label>
          <input
            type='text'
            value={firstName}
            onChange={(e) => setFirstName(e.target.value)}
            className='border-2 border-gray-500 px-4 py-2  w-full '
          />
        </div>
        <div className='my-4'>
          <label className='text-xl mr-4 text-gray-500'>Last Name</label>
          <input
            type='text'
            value={lastName}
            onChange={(e) => setLastName(e.target.value)}
            className='border-2 border-gray-500 px-4 py-2  w-full '
          />
        </div>
        <div className='my-4'>
          <label className='text-xl mr-4 text-gray-500'>Age</label>
          <input
            type='number'
            value={age}
            onChange={(e) => setAge(e.target.value)}
            className='border-2 border-gray-500 px-4 py-2  w-full '
          />
        </div>
        <div className='my-4'>
          <label className='text-xl mr-4 text-gray-500'>Team</label>
          <input
            type='text'
            value={team}
            onChange={(e) => setTeam(e.target.value)}
            className='border-2 border-gray-500 px-4 py-2  w-full '
          />
        </div>
        <div className='my-4'>
          <label className='text-xl mr-4 text-gray-500'>Position</label>
          <input
            type='text'
            value={position}
            onChange={(e) => setPosition(e.target.value)}
            className='border-2 border-gray-500 px-4 py-2  w-full '
          />
        </div>
        <div className='my-4'>
          <label className='text-xl mr-4 text-gray-500'>DOB</label>
          <input
            type='date'
            value={DOB}
            onChange={(e) => setDOB(e.target.value)}
            className='border-2 border-gray-500 px-4 py-2  w-full '
          />
        </div>
        <div className='my-4'>
          <label className='text-xl mr-4 text-gray-500'>sport</label>
          <input
            type='text'
            value={sport}
            onChange={(e) => setSport(e.target.value)}
            className='border-2 border-gray-500 px-4 py-2  w-full '
          />
        </div>
        <button className='p-2 bg-sky-300 m-8' onClick={handleUpdatePlayer}>
          Save
        </button>
      </div>
    </div>
  );
};

export default UpdatePlayer;
