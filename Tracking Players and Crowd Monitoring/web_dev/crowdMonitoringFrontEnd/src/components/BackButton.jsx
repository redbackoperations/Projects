import React from 'react'
import { Link } from 'react-router-dom'
import {BsArrowLeft} from 'react-icons/bs'

const BackButton = ({destinaion = '/players'}) => { 
  return (
    <div className='flex'>
       <Link to= {destinaion} className='bg-sky-800 text-white px-4 py-1 rounded-lg w-fit'>
        <BsArrowLeft className='tesxt-2x1' />
       </Link>
    </div>
  )
}

export default BackButton
