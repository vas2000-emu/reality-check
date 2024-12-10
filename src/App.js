import React, { useState } from 'react';

function FileUpload({ setPreviewSource, setCoveragePercentage, setPrediction, setInfoBoxVisible }) {
    const [selectedFile, setSelectedFile] = useState(null);
    const [linkClicked, setLinkClicked] = useState(false);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file && file.type.startsWith('image/')) {
            setSelectedFile(file);
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreviewSource(reader.result);
            };
            reader.readAsDataURL(file);
            uploadImage(file);
            setInfoBoxVisible(true);
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
                setCoveragePercentage(percentage);
                setPrediction(aiProbability > 0.5 ? 'AI-generated' : 'Real');
            } else {
                alert('Failed to upload the image.');
            }
        } catch (error) {
            console.error('Error uploading image:', error);
            alert('An error occurred while uploading the image.');
        }
    };

    const handleLinkClick = () => {
        setLinkClicked(true);
        window.open('https://drive.google.com/drive/folders/1BCWRh13DqZGT-DcalODgCBNqm0kTZzMu?usp=sharing', '_blank');
    };

    return (
        <div style={styles.fileUploadContainer}>
            {!linkClicked ? (
                <button
                    onClick={handleLinkClick}
                    style={styles.uploadButton}
                >
                    BEGIN HERE
                </button>
            ) : (
                <button
                    onClick={() => document.getElementById('file-input').click()}
                    style={styles.uploadButton}
                >
                    {selectedFile ? 'Replace File' : 'Upload Image Here!'}
                </button>
            )}
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
    const [infoBoxVisible, setInfoBoxVisible] = useState(false);

    return (
        <div style={styles.gradient}>
            <div style={styles.container}>
                <div style={styles.fileUploadArea}>
                    <h1 style={styles.logoText}>reAlity|check</h1>
                    <FileUpload
                        setPreviewSource={setPreviewSource}
                        setCoveragePercentage={setCoveragePercentage}
                        setPrediction={setPrediction}
                        setInfoBoxVisible={setInfoBoxVisible}
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
                {infoBoxVisible && (
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
        marginBottom: '10px',
    },
    imageContainer: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: 'fit-content',
    },
    imagePreview: {
        width: '400px',
        height: '400px',
        objectFit: 'cover',
        borderRadius: '10px',
        marginTop: '20px',
    },
    infoBox: {
        position: 'fixed',
        right: '3%',
        top: '50%',
        transform: 'translateY(-50%)',
        backgroundColor: 'rgba(255, 255, 255, 0.8)',
        padding: '20px',
        borderRadius: '10px',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.2)',
        textAlign: 'center',
        transition: 'opacity 0.5s',
        opacity: 1,
    },
    infoContent: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-start',
    },
    percentage: {
        fontSize: '48px',
        fontWeight: 'bold',
        color: '#000',
        marginRight: '10px',
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
        textAlign: 'left',
    },
};
