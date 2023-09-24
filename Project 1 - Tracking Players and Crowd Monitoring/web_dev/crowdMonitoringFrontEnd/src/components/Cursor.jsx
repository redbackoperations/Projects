import React, { useState, useEffect } from 'react';
import './Cursor.css'; // Import your CSS stylesheet for styling

const Cursor = () => {
  const [position, setPosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e) => {
      setPosition({ x: e.clientX, y: e.clientY });
    };

    window.addEventListener('mousemove', handleMouseMove);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  return (
    <div className="cool-cursor" style={{ left: `${position.x}px`, top: `${position.y}px` }}>
      <div className="inner-cursor"></div>
    </div>
  );
};

export default Cursor;
