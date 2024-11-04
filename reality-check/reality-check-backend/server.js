const express = require('express');
const multer = require('multer');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5000;

// Configure multer
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/'); // specify the upload directory
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname); // use original file name
    },
});

const upload = multer({ storage });

app.post('/upload', upload.single('file'), (req, res) => {
    console.log('Request body:', req.body);
    console.log('Uploaded file:', req.file);
    // Check if file is available
    if (!req.file) {
        return res.status(400).send('No file uploaded.');
    }
    console.log('File uploaded successfully:', req.file.path);
    res.status(200).send({ path: req.file.path }); // send back the file path
});

app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('Something broke!');
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
