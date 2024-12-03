import React, { useState } from 'react';

function FileUpload({ setPreviewSource, setCoveragePercentage }) {
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
        } else {
            alert('Please upload a valid image file.');
        }
    };

    // Function to send the image file to the backend
    const uploadImage = async (file) => {
        const formData = new FormData();
        formData.append('image', file); // Append the image file to FormData

        try {
            // Replace with the actual backend API endpoint
            const response = await fetch('/api/predict-ai-image', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                const { coveragePercentage } = data; // Assuming the backend returns the coverage percentage
                setCoveragePercentage(coveragePercentage); // Update the coverage percentage in the parent component
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
                style={{ display: 'none' }}
                accept="image/*"
            />
        </div>
    );
}


export default function App() {
    const [previewSource, setPreviewSource] = useState(null);
    const [coveragePercentage, setCoveragePercentage] = useState(0); // Default to 0%
    const [percentageTextVisible, setPercentageTextVisible] = useState(false); // Controls visibility of the percentage text

    return (
        <div style={styles.gradient}>
            <div style={styles.container}>
                <div style={styles.fileUploadArea}>
                    <h1 style={styles.logoText}>reAlity|check</h1>
                    <FileUpload
                        setPreviewSource={setPreviewSource}
                        setCoveragePercentage={setCoveragePercentage}
                    />
                </div>
                {previewSource && (
                    <div style={styles.imageContainer}>
                        <img
                            src={previewSource}
                            alt="Preview"
                            style={styles.imagePreview}
                        />
                        <div
                            style={{
                                ...styles.overlay,
                                height: `${coveragePercentage}%`,
                                bottom: 0,
                                top: 'auto',
                                transition: 'height 1s ease-in-out',
                                borderBottomLeftRadius: '10px',
                                borderBottomRightRadius: '10px',
                            }}
                            onTransitionEnd={() => setPercentageTextVisible(true)}
                            data-testid="overlay" // Add data-testid for easy testing
                        />
                        {percentageTextVisible && (
                            <div style={styles.percentageText} data-testid="percentage-text">
                                {coveragePercentage}% Coverage
                            </div>
                        )}
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
        position: 'relative',
        width: '80%',
        maxWidth: '800px',
    },
    imagePreview: {
        width: '100%',
        borderRadius: '10px',
    },
    overlay: {
        position: 'absolute',
        left: '0',
        width: '100%',
        backgroundColor: 'rgba(255, 255, 255, 0.25)',
        zIndex: '1',
        borderBottomLeftRadius: '10px', // Round bottom left corner
        borderBottomRightRadius: '10px', // Round bottom right corner
    },
    percentageText: {
        position: 'absolute',
        bottom: '10px',
        left: '50%',
        transform: 'translateX(-50%)',
        fontSize: '20px',
        color: '#fff',
        fontWeight: 'bold',
    },
};
