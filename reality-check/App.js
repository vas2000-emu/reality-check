import React, { useState } from 'react';
import { Text, View, StyleSheet, TouchableOpacity, Image, Dimensions } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

// Get the device's screen dimensions
const { height: screenHeight, width: screenWidth } = Dimensions.get('window');

function FileUpload({ setPreviewSource }) {
    const [selectedFile, setSelectedFile] = useState(null);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file && (file.type === 'image/jpeg' || file.type === 'image/jpg')) {
            setSelectedFile(file);
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreviewSource(reader.result);
            };
            reader.readAsDataURL(file);
        } else {
            alert('Please upload a valid JPG file.');
        }
    };

    return (
        <View style={styles.fileUploadContainer}>
            <TouchableOpacity
                onPress={() => document.getElementById('file-input').click()}
                style={styles.uploadButton}
            >
                <Text style={styles.buttonText}>{selectedFile ? 'Replace File' : 'Upload Image Here!'}</Text>
            </TouchableOpacity>
            <input
                type="file"
                id="file-input"
                onChange={handleFileChange}
                style={styles.fileInput}
                accept=".jpg,.jpeg"
            />
        </View>
    );
}

export default function App() {
    const [previewSource, setPreviewSource] = useState(null);

    // Calculate the height of the image preview based on the screen size minus padding
    const imagePreviewHeight = screenHeight - 40;

    return (
        <LinearGradient colors={['#001f3f', '#0074D9']} style={styles.gradient}>
            <View style={styles.container}>
                <View style={styles.fileUploadArea}>
                    <Text style={styles.text}>reAlity|check</Text>
                    <FileUpload setPreviewSource={setPreviewSource} />
                </View>
                {previewSource && (
                    <Image
                        source={{ uri: previewSource }}
                        style={[styles.imagePreview, { height: imagePreviewHeight }]}
                    />
                )}
            </View>
        </LinearGradient>
    );
}

const styles = StyleSheet.create({
    gradient: {
        flex: 1,
        flexDirection: 'row',
    },
    container: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center', // Centering all items in the container
        paddingTop: 20,
        paddingBottom: 20,
        paddingHorizontal: 20,
    },
    fileUploadArea: {
        width: '20%', // Set width to 20% of the screen
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'rgba(255, 255, 255, 0.8)', // Slight background for contrast
        borderRadius: 10,
        padding: 10,
        shadowColor: '#000',
        shadowOffset: {
            width: 0,
            height: 1,
        },
        shadowOpacity: 0.2,
        shadowRadius: 1.41,
        elevation: 2,
    },
    text: {
        fontSize: 24,
        color: '#000',
        marginBottom: 20,
    },
    fileUploadContainer: {
        alignItems: 'center',
        justifyContent: 'flex-end',
    },
    fileInput: {
        display: 'none', // Hide the file input
    },
    uploadButton: {
        backgroundColor: '#0074D9',
        padding: 10,
        borderRadius: 5,
        marginBottom: 10,
        width: '100%', // Full width for button
        alignItems: 'center',
    },
    buttonText: {
        color: '#fff',
        fontSize: 16,
    },
    imagePreview: {
        width: '60%', // Occupy 60% of the screen width
        borderRadius: 10,
        resizeMode: 'cover',
        marginLeft: 10,
    },
});
