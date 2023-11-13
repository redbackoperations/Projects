import React, { useState } from 'react';
import BackButton from '../components/BackButton.jsx';
import Spinner from '../components/Spinner';
import axios from 'axios';
import { Navigate, useNavigate } from 'react-router-dom';

const CreatePlayer = () => {
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [age, setAge] = useState('');
  const [team, setTeam] = useState('');
  const [position, setPosition] = useState('');
  const [dateOfBirth, setDateOfBirth] = useState('');
  const [sport, setSport] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleSavePlayer = () => {
    setError(null);
    const data = {
      firstName,
      lastName,
      age,
      team,
      position,
      dateOfBirth,
      sport,
    };

    if (
        !firstName ||
        !lastName ||
        !age ||
        !team ||
        !position ||
        !dateOfBirth ||
        !sport
    ) {
        setError('Please enter all the fields.');
        return;
    }


    setLoading(true);
    axios
      .post('http://localhost:5004/players', data)
      .then((res) => {
        setLoading(false);
        navigate('/players'); 
      })
      .catch((err) => {
        setLoading(false);
        console.log(err);
        setError('An error occurred while saving the player.');
      });
  };

  return (
    <div className="p-4">
      <BackButton />
      <h1 className="text-3xl my-4">Create New Player</h1>
      {loading ? <Spinner /> : ''}
      <div className="flex flex-col border-2 border-sky-400 rounded-xl w-[600px] p-4 mx-auto">
        <div className="my-4">
          <label className="text-xl mr-4 text-gray-500">First Name</label>
          <input
            type="text"
            value={firstName}
            onChange={(e) => setFirstName(e.target.value)}
            className="border-2 border-gray-500 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-sky-600 focus:border-transparent"
          />
        </div>
        <div className="my-4">
          <label className="text-xl mr-4 text-gray-500">Last Name</label>
          <input
            type="text"
            value={lastName}
            onChange={(e) => setLastName(e.target.value)}
            className="border-2 border-gray-500 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-sky-600 focus:border-transparent"
          />
        </div>
        <div className="my-4">
          <label className="text-xl mr-4 text-gray-500">Age</label>
          <input
            type="text"
            value={age}
            onChange={(e) => setAge(e.target.value)}
            className="border-2 border-gray-500 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-sky-600 focus:border-transparent"
          />
        </div>
        <div className="my-4">
          <label className="text-xl mr-4 text-gray-500">Team</label>
          <input
            type="text"
            value={team}
            onChange={(e) => setTeam(e.target.value)}
            className="border-2 border-gray-500 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-sky-600 focus:border-transparent"
          />
        </div>
        <div className="my-4">
          <label className="text-xl mr-4 text-gray-500">Position</label>
          <input
            type="text"
            value={position}
            onChange={(e) => setPosition(e.target.value)}
            className="border-2 border-gray-500 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-sky-600 focus:border-transparent"
          />
        </div>
        <div className="my-4">
            <label className="text-xl mr-4 text-gray-500">Date of Birth</label>
            <input
                type="date" // Use the 'date' type for DOB input
                value={dateOfBirth}
                onChange={(e) => setDateOfBirth(e.target.value)}
                className="border-2 border-gray-500 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-sky-600 focus:border-transparent"
                placeholder="YYYY-MM-DD" // Optionally add a placeholder
            />
        </div>

        <div className="my-4">
          <label className="text-xl mr-4 text-gray-500">Sport</label>
          <input
            type="text"
            value={sport}
            onChange={(e) => setSport(e.target.value)}
            className="border-2 border-gray-500 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-sky-600 focus:border-transparent"
          />
        </div>
        <button className="p-2 bg-sky-300 m-8" onClick={handleSavePlayer}>
          Save
        </button>
      </div>
      {error && <p className="text-red-500">{error}</p>}
    </div>
  );
};

export default CreatePlayer;
