import React, { useState } from 'react';

function FileUpload({ setPreviewSource, setCoveragePercentage, setPrediction, setImageUploaded }) {
    const [selectedFile, setSelectedFile] = useState(null);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file && file.type.startsWith('image/')) {
            setSelectedFile(file);
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreviewSource(reader.result);
            };
            reader.readAsDataURL(file);

            // Send the image file to the backend after it's selected
            uploadImage(file);
            setImageUploaded(true);
        } else {
            alert('Please upload a valid image file.');
        }
    };

    const uploadImage = async (file) => {
        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {  // Update this URL with your deployed Flask API endpoint
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                const aiProbability = data.percent_ai;
                let percentage = (aiProbability * 100).toFixed(2);
                if (aiProbability <= 0.5) {
                    percentage = (100 - percentage).toFixed(2);
                }
                setCoveragePercentage(percentage);  // Set the coverage percentage
                setPrediction(aiProbability > 0.5 ? 'AI-generated' : 'Real');
            } else {
                alert('Failed to upload the image.');
            }
        } catch (error) {
            console.error('Error uploading image:', error);
            alert('An error occurred while uploading the image.');
        }
    };

    return (
        <div style={styles.fileUploadContainer}>
            <button
                onClick={() => document.getElementById('file-input').click()}
                style={styles.uploadButton}
            >
                {selectedFile ? 'Replace File' : 'Upload Image Here!'}
            </button>
            <input
                type="file"
                id="file-input"
                onChange={handleFileChange}
                onClick={(event) => event.target.value = null}
                style={{ display: 'none' }}
                accept="image/*"
            />
        </div>
    );
}

export default function App() {
    const [previewSource, setPreviewSource] = useState(null);
    const [coveragePercentage, setCoveragePercentage] = useState(0);
    const [prediction, setPrediction] = useState('');
    const [imageUploaded, setImageUploaded] = useState(false);

    return (
        <div style={styles.gradient}>
            <div style={styles.container}>
                <div style={styles.fileUploadArea}>
                    <h1 style={styles.logoText}>reAlity|check</h1>
                    <FileUpload
                        setPreviewSource={setPreviewSource}
                        setCoveragePercentage={setCoveragePercentage}
                        setPrediction={setPrediction}
                        setImageUploaded={setImageUploaded}
                    />
                </div>
                {previewSource && (
                    <div style={styles.imageContainer}>
                        <img
                            src={previewSource}
                            alt="Preview"
                            style={styles.imagePreview}
                        />
                    </div>
                )}
                {imageUploaded && (
                    <div style={styles.infoBox}>
                        <div style={styles.infoContent}>
                            <span style={styles.percentage}>{coveragePercentage}%</span>
                            <span style={styles.predictionText}>{prediction}</span>
                        </div>
                        <p style={styles.probabilityText}>Probability</p>
                    </div>
                )}
            </div>
        </div>
    );
}

const styles = {
    gradient: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        width: '100vw',
        background: 'linear-gradient(to bottom, #001f3f, #0074D9)',
        padding: '0',
        margin: '0',
    },
    container: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '20px',
        width: '100%',
        height: '100%',
    },
    fileUploadArea: {
        textAlign: 'center',
        backgroundColor: 'rgba(255, 255, 255, 0.8)',
        borderRadius: '10px',
        padding: '20px',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.2)',
    },
    logoText: {
        fontSize: '36px',
        color: '#000',
        fontWeight: 'bold',
        fontFamily: `'Courier New', Courier, monospace`,
        marginBottom: '20px',
    },
    fileUploadContainer: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
    },
    uploadButton: {
        backgroundColor: '#0074D9',
        color: '#fff',
        padding: '10px 20px',
        borderRadius: '5px',
        cursor: 'pointer',
        border: 'none',
    },
    imageContainer: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: 'fit-content',
    },
    imagePreview: {
        width: '100%',  // Adjust width to match upload box width
        maxWidth: '400px',  // Set a maximum width to control the image size
        borderRadius: '10px',
        marginTop: '20px',
    },
    infoBox: {
        position: 'fixed',
        right: '5%',
        top: '50%',
        transform: 'translateY(-50%)',
        backgroundColor: 'rgba(255, 255, 255, 0.8)',
        padding: '20px',
        borderRadius: '10px',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.2)',
        textAlign: 'center',
    },
    infoContent: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    },
    percentage: {
        fontSize: '48px',
        fontWeight: 'bold',
        color: '#000',
        marginRight: '10px',  // Add some space between the percentage and the prediction text
    },
    predictionText: {
        fontSize: '48px',
        fontWeight: 'bold',
        color: '#000',
    },
    probabilityText: {
        fontSize: '24px',
        color: '#000',
        marginTop: '10px',
    },
};
