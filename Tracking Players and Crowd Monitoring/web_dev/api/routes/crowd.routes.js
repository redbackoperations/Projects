/**
 * Express API routes for managing player data.
 * @module routes/crowdRoutes
 */

import express from 'express';
import {Crowd} from '../models/crowd.model.js';

const router = express.Router();

/**
 * Get all crowd data records.
 * @route GET /api/crowd
 * @group Crowd - Operations related to crowd data
 * @returns {Array.<Object>} 200 - An array of crowd data records
 * @throws {Error} 500 - Server error
 */

router.get('/', async (req, res) => {
  try {
    const crowdData = await Crowd.find();
    res.status(200).json(crowdData);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

/**
 * Get all crowd heat map data records.
 * @route GET /api/crowd/heat-map
 * @group Crowd - Operations related to crowd data
 * @returns {Array.<Object>} 200 - An array of crowd heat map data records
 * @throws {Error} 500 - Server error
 */
router.get('/heat-map', async (req, res) => {
  try {
    const crowdHeatMapData = await Crowd.find().select('heatMapData');
    res.status(200).json(crowdHeatMapData);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

/**
 * Get all crowd events or incidents data records.
 * @route GET /api/crowd/eventsOrIncidents
 * @group Crowd - Operations related to crowd data
 * @returns {Array.<Object>} 200 - An array of crowd events or incidents data records
 * @throws {Error} 500 - Server error
 */
router.get('/eventsOrIncidents', async (req, res) => {
  try {
    // Assuming events or incidents data is stored as a string in the database
    const crowdEventsOrIncidents = await Crowd.find().select('eventsOrIncidents');
    res.status(200).json(crowdEventsOrIncidents);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

/**
 * Get all crowd trends data records.
 * @route GET /api/crowd/trends
 * @group Crowd - Operations related to crowd data
 * @returns {Array.<Object>} 200 - An array of crowd trends data records
 * @throws {Error} 500 - Server error
 */
router.get('/trends', async (req, res) => {
  try {
    const crowdTrends = await Crowd.find().select('trends');
    res.status(200).json(crowdTrends);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});
/**
 * Create a new crowd data record.
 * @route POST /api/crowd
 * @group Crowd - Operations related to crowd data
 * @param {string} timestamp.body.required - Timestamp of the crowd data
 * @param {string} location.body.required - Location of the crowd data
 * @param {number} density.body.required - Crowd density
 * @param {string} heatMapData.body.required - Heat map data of the crowd
 * @param {string} eventsOrIncidents.body.required - Events or incidents data of the crowd
 * @param {string} trends.body.required - Trends data of the crowd
 * @returns {Object} 201 - The created crowd data record
 * @throws {Error} 400 - Missing parameters or invalid input
 * @throws {Error} 500 - Server error
 */
router.post('/', async (req, res) => {
    try {
        if (
            !req.body.timestamp ||
            !req.body.location ||
            !req.body.density ||
            !req.body.heatMapData ||
            !req.body.eventsOrIncidents ||
            !req.body.trends
        ) {
            return res.status(400).send({ message: "Missing parameters" });
        }
      const CrowdData = {
        timestamp: req.body.timestamp,
        location: req.body.location,
        density: req.body.density,
        heatMapData: req.body.heatMapData,
        eventsOrIncidents: req.body.eventsOrIncidents,
        trends: req.body.trends,
      };

      const newCrowdData = await Crowd.create(CrowdData);
  
      res.status(201).json(newCrowdData);
    } catch (error) {
      res.status(500).json({ message: error.message });
    }
  });

/**
 * Update crowd heat map data by location.
 * @route POST /api/crowd/heat-map
 * @group Crowd - Operations related to crowd data
 * @param {string} location.body.required - Location of the crowd data
 * @param {string} heatMapData.body.required - Updated heat map data
 * @returns {Object} 200 - Updated crowd data record with new heat map data
 * @throws {Error} 400 - Missing parameters or invalid input
 * @throws {Error} 404 - Crowd record not found
 * @throws {Error} 500 - Server error
 */
router.post('/heat-map', async (req, res) => {
try {
    if (!req.body.location || !req.body.heatMapData) {
    return res.status(400).send({ message: 'Missing parameters (enter heat map data and location)' });
    }

    const crowdRecord = await Crowd.findOne({ location: req.body.location });

    if (!crowdRecord) {
    return res.status(404).json({ message: 'Crowd record not found' });
    }


    crowdRecord.heatMapData = req.body.heatMapData;


    await crowdRecord.save();

    res.status(200).json(crowdRecord);
} catch (error) {
    res.status(500).json({ message: error.message });
}
});

/**
 * Update crowd events or incidents data by location.
 * @route POST /api/crowd/eventsOrIncidents
 * @group Crowd - Operations related to crowd data
 * @param {string} location.body.required - Location of the crowd data
 * @param {string} eventsOrIncidents.body.required - Updated events or incidents data
 * @returns {Object} 200 - Updated crowd data record with new events or incidents data
 * @throws {Error} 400 - Missing parameters or invalid input
 * @throws {Error} 404 - Crowd record not found
 * @throws {Error} 500 - Server error
 */

router.post('/eventsOrIncidents', async (req, res) => {
  try {
    if (!req.body.location || !req.body.eventsOrIncidents) {
      return res.status(400).send({ message: 'Missing parameters' });
    }

    const crowdRecord = await Crowd.findOne({ location: req.body.location });

    if (!crowdRecord) {
      return res.status(404).json({ message: 'Crowd record not found' });
    }


    crowdRecord.eventsOrIncidents = req.body.eventsOrIncidents;


    await crowdRecord.save();

    res.status(200).json(crowdRecord);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

/**
 * Update crowd trends data by location.
 * @route POST /api/crowd/trends
 * @group Crowd - Operations related to crowd data
 * @param {string} location.body.required - Location of the crowd data
 * @param {string} trends.body.required - Updated trends data
 * @returns {Object} 200 - Updated crowd data record with new trends data
 * @throws {Error} 400 - Missing parameters or invalid input
 * @throws {Error} 404 - Crowd record not found
 * @throws {Error} 500 - Server error
 */
router.post('/trends', async (req, res) => {
  try {
    if (!req.body.location || !req.body.trends) {
      return res.status(400).send({ message: 'Missing parameters' });
    }


    const crowdRecord = await Crowd.findOne({ location: req.body.location });

    if (!crowdRecord) {
      return res.status(404).json({ message: 'Crowd record not found' });
    }


    crowdRecord.trends = req.body.trends;


    await crowdRecord.save();

    res.status(200).json(crowdRecord);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

/**
 * Update crowd data by ID.
 * @route PUT /api/crowd/{id}
 * @group Crowd - Operations related to crowd data
 * @param {string} id.path.required - Crowd data ID
 * @param {string} timestamp.body.required - Updated timestamp
 * @param {string} location.body.required - Updated location
 * @param {number} density.body.required - Updated crowd density
 * @param {string} heatMapData.body.required - Updated heat map data
 * @param {string} eventsOrIncidents.body.required - Updated events or incidents data
 * @param {string} trends.body.required - Updated trends data
 * @returns {Object} 200 - Crowd data updated successfully
 * @throws {Error} 400 - Missing parameters or invalid input
 * @throws {Error} 404 - Crowd record not found
 * @throws {Error} 500 - Server error
 */
router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const updatedData = req.body;
    if (
      !updatedData.timestamp ||
      !updatedData.location ||
      !updatedData.density ||
      !updatedData.heatMapData ||
      !updatedData.eventsOrIncidents ||
      !updatedData.trends
    ) {
      return res.status(400).send({ message: 'Missing parameters' });
    }

    const crowdRecord = await Crowd.findById(id);

    if (!crowdRecord) {
      return res.status(404).json({ message: 'Crowd record not found' });
    }

    const result = await Object.assign(crowdRecord, updatedData);

    if (!result) {
      return res.status(500).json({ message: 'Error updating crowd record' });
    }

    await crowdRecord.save(); // Save the updated crowd record

    res.status(200).json({ message: 'Data updated successfully' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

/**
 * Update crowd heat map data by location.
 * @route PUT /api/crowd/heat-map/{id}
 * @group Crowd - Operations related to crowd data
 * @param {string} id.path.required - Crowd data ID
 * @param {string} heatMapData.body.required - Updated heat map data
 * @returns {Object} 200 - Updated crowd data record with new heat map data
 * @throws {Error} 400 - Missing parameters or invalid input
 * @throws {Error} 404 - Crowd record not found
 * @throws {Error} 500 - Server error
 */
router.put('/heat-map/:id', async (req, res) => {
  try {
    const crowdId = req.params.id;
    const updatedHeatMapData = req.body.heatMapData;

    if (!req.body.heatMapData) {
      return res.status(400).send({ message: 'Missing parameters' });
    }


    const crowdRecord = await Crowd.findById(crowdId);

    if (!crowdRecord) {
      return res.status(404).json({ message: 'Crowd record not found' });
    }


    crowdRecord.heatMapData = updatedHeatMapData;


    await crowdRecord.save();

    res.status(200).json(crowdRecord);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

/**
 * Update crowd events or incidents data by location.
 * @route PUT /api/crowd/eventsOrIncidents/{id}
 * @group Crowd - Operations related to crowd data
 * @param {string} id.path.required - Crowd data ID
 * @param {string} eventsOrIncidents.body.required - Updated events or incidents data
 * @returns {Object} 200 - Updated crowd data record with new events or incidents data
 * @throws {Error} 400 - Missing parameters or invalid input
 * @throws {Error} 404 - Crowd record not found
 * @throws {Error} 500 - Server error
 */
router.put('/eventsOrIncidents/:id', async (req, res) => {
  try {
    const crowdId = req.params.id;
    const updatedEventsOrIncidents = req.body.eventsOrIncidents;

    if (!req.body.eventsOrIncidents) {
      return res.status(400).send({ message: 'Missing parameters' });
    }
    const crowdRecord = await Crowd.findById(crowdId);

    if (!crowdRecord) {
      return res.status(404).json({ message: 'Crowd record not found' });
    }


    crowdRecord.eventsOrIncidents = updatedEventsOrIncidents;


    await crowdRecord.save();

    res.status(200).json(crowdRecord);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

/**
 * Update crowd trends data by location.
 * @route PUT /api/crowd/trends/{id}
 * @group Crowd - Operations related to crowd data
 * @param {string} id.path.required - Crowd data ID
 * @param {string} trends.body.required - Updated trends data
 * @returns {Object} 200 - Updated crowd data record with new trends data
 * @throws {Error} 400 - Missing parameters or invalid input
 * @throws {Error} 404 - Crowd record not found
 * @throws {Error} 500 - Server error
 */
router.put('/trends/:id', async (req, res) => {
  try {
    const crowdId = req.params.id;
    const updatedTrends = req.body.trends;


    const crowdRecord = await Crowd.findById(crowdId);

    if (!crowdRecord) {
      return res.status(404).json({ message: 'Crowd record not found' });
    }


    crowdRecord.trends = updatedTrends;


    await crowdRecord.save();

    res.status(200).json(crowdRecord);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

export default router;
