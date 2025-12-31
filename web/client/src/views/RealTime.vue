<script setup>
import { ref } from "vue";

const apiUrl = import.meta.env.VITE_BASE_URL || "http://127.0.0.1:8000";
const selectedExercise = ref(null);
const isStreaming = ref(false);

const startStream = (exercise) => {
    selectedExercise.value = exercise;
    isStreaming.value = true;
};

const stopStream = () => {
    isStreaming.value = false;
    selectedExercise.value = null;
    // Force reload the image to stop the stream connection
    window.stop();
};

const getStreamUrl = () => {
    if (!selectedExercise.value) return "";
    return `${apiUrl}/api/video/feed?type=${selectedExercise.value}`;
};
</script>

<template>
    <div class="realtime-container">
        <h1>Real Time Analysis</h1>
        
        <div class="controls" v-if="!isStreaming">
            <p>Select an exercise to start:</p>
            <div class="buttons">
                <button @click="startStream('squat')">Squat</button>
                <button @click="startStream('plank')">Plank</button>
                <button @click="startStream('bicep_curl')">Bicep Curl</button>
            </div>
        </div>

        <div class="stream-wrapper" v-if="isStreaming">
            <div class="header">
                <h2>{{ selectedExercise }} Analysis</h2>
                <button class="stop-btn" @click="stopStream">Stop</button>
            </div>
            <img :src="getStreamUrl()" alt="Live Stream" class="video-feed" />
        </div>
    </div>
</template>

<style lang="scss" scoped>
.realtime-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2rem;
    margin-top: 2rem;
    width: 100%;

    h1 {
        color: var(--primary-color);
        text-transform: uppercase;
    }

    .controls {
        text-align: center;
        
        p {
            margin-bottom: 1rem;
            color: var(--secondary-color);
            font-size: 1.2rem;
        }

        .buttons {
            display: flex;
            gap: 1rem;

            button {
                padding: 1rem 2rem;
                font-size: 1.2rem;
                background-color: var(--primary-color);
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: transform 0.2s;

                &:hover {
                    transform: scale(1.05);
                }
            }
        }
    }

    .stream-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1rem;
        width: 80%;
        max-width: 800px;

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;

            h2 {
                text-transform: capitalize;
                color: var(--secondary-color);
            }

            .stop-btn {
                padding: 0.5rem 1rem;
                background-color: #ff4444;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
        }

        .video-feed {
            width: 100%;
            border: 4px solid var(--primary-color);
            border-radius: 8px;
        }
    }
}
</style>
