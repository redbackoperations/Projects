import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Spinner from '../components/Spinner';
import Cursor from '../components/cursor';
import { Link } from 'react-router-dom';
import { AiOutlineEdit } from 'react-icons/ai';
import { BsInfoCircle } from 'react-icons/bs';
import { MdOutlineAddBox, MdOutlineDelete } from 'react-icons/md';
import { RiPlayFill } from 'react-icons/ri'; // Import an icon for navigating to player movements

const ListPlayers = () => {
  const [players, setPlayers] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    axios
      .get('http://localhost:5004/players')
      .then((response) => {
        setPlayers(response.data.data);
        setLoading(false);
      })
      .catch((error) => {
        console.log(error);
        setLoading(false);
      });
  }, []);

  return (
    <div className='p-4'>
      <div className='flex justify-between items-center'>
        <h1 className='text-3xl my-8 text-center font-semibold text-indigo-700'>
          Player List
        </h1>
        <Link to='/player/create'>
          <MdOutlineAddBox className='text-sky-800 text-4xl' />
        </Link>
      </div>
      {loading ? (
        <Spinner />
      ) : (
        <table className='w-full border-collapse rounded-lg overflow-hidden shadow-lg'>
          <thead>
            <tr>
              <th className='border p-3 text-xl bg-indigo-200 text-indigo-800 rounded-tl-lg'>
                No
              </th>
              <th className='border p-3 text-xl bg-indigo-200 text-indigo-800'>
                Player Name
              </th>
              <th className='border p-3 text-xl bg-indigo-200 text-indigo-800'>
                Team
              </th>
              <th className='border p-3 text-xl bg-indigo-200 text-indigo-800'>
                Position
              </th>
              <th className='border p-3 text-xl bg-indigo-200 text-indigo-800'>
                Date of Birth
              </th>
              <th className='border p-3 text-xl bg-indigo-200 text-indigo-800'>
                Sport
              </th>
              <th className='border p-3 text-xl bg-indigo-200 text-indigo-800'>
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {players.map((player, index) => (
              <tr
                key={player._id}
                className={`h-12 ${
                  index % 2 === 0 ? 'bg-gray-100' : 'bg-gray-200'
                }`}
              >
                <td className='border p-3 text-center'>{index + 1}</td>
                <td className='border p-3'>{`${player.firstName} ${player.lastName}`}</td>
                <td className='border p-3'>{player.team}</td>
                <td className='border p-3'>{player.position}</td>
                <td className='border p-3'>{player.dateOfBirth}</td>
                <td className='border p-3'>{player.sport}</td>
                <td className='border p-3 text-center'>
                  <div className='flex justify-around gap-4'>
                    <Link
                      to={`/player/${player._id}`}
                      className='text-indigo-600 hover:text-indigo-800 transition duration-300'
                    >
                      <BsInfoCircle className='text-2xl' />
                    </Link>
                    <Link
                      to={`/player/update/${player._id}`}
                      className='text-yellow-600 hover:text-yellow-800 transition duration-300'
                    >
                      <AiOutlineEdit className='text-2xl' />
                    </Link>
                    <Link
                      to={`/player/delete/${player._id}`}
                      className='text-red-600 hover:text-red-800 transition duration-300'
                    >
                      <MdOutlineDelete className='text-2xl' />
                    </Link>
                    <Link
                      to={`/player/movements/${player._id}`} // Link to player movements
                      className='text-green-600 hover:text-green-800 transition duration-300'
                    >
                      <RiPlayFill className='text-2xl' />
                    </Link>
                    <Link
                      to={`/player/update/stats/${player._id}`} // Link to player statistics update
                      className='text-orange-600 hover:text-orange-800 transition duration-300'
                      >
                      <MdOutlineAddBox className='text-2xl' />
                    </Link>
                    <Cursor />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default ListPlayers;
