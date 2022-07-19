{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:41:58.064064Z",
     "iopub.status.busy": "2022-07-19T10:41:58.063587Z",
     "iopub.status.idle": "2022-07-19T10:42:00.531720Z",
     "shell.execute_reply": "2022-07-19T10:42:00.530266Z",
     "shell.execute_reply.started": "2022-07-19T10:41:58.063979Z"
    }
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import librosa\n",
    "import librosa.display\n",
    "from IPython.display import Audio\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:42:11.107510Z",
     "iopub.status.busy": "2022-07-19T10:42:11.107108Z",
     "iopub.status.idle": "2022-07-19T10:42:13.237039Z",
     "shell.execute_reply": "2022-07-19T10:42:13.235407Z",
     "shell.execute_reply.started": "2022-07-19T10:42:11.107479Z"
    }
   },
   "outputs": [],
   "source": [
    "paths = []\n",
    "labels = []\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        paths.append(os.path.join(dirname, filename))\n",
    "        label = filename.split('_')[-1]\n",
    "        label = label.split('.')[0]\n",
    "        labels.append(label.lower())\n",
    "print('Dataset is Loaded')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:42:22.447259Z",
     "iopub.status.busy": "2022-07-19T10:42:22.446795Z",
     "iopub.status.idle": "2022-07-19T10:42:22.461457Z",
     "shell.execute_reply": "2022-07-19T10:42:22.460012Z",
     "shell.execute_reply.started": "2022-07-19T10:42:22.447227Z"
    }
   },
   "outputs": [],
   "source": [
    "paths[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:42:28.549963Z",
     "iopub.status.busy": "2022-07-19T10:42:28.549578Z",
     "iopub.status.idle": "2022-07-19T10:42:28.558800Z",
     "shell.execute_reply": "2022-07-19T10:42:28.557480Z",
     "shell.execute_reply.started": "2022-07-19T10:42:28.549934Z"
    }
   },
   "outputs": [],
   "source": [
    "labels[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:42:36.402403Z",
     "iopub.status.busy": "2022-07-19T10:42:36.401819Z",
     "iopub.status.idle": "2022-07-19T10:42:36.439656Z",
     "shell.execute_reply": "2022-07-19T10:42:36.438374Z",
     "shell.execute_reply.started": "2022-07-19T10:42:36.402365Z"
    }
   },
   "outputs": [],
   "source": [
    "## Create a dataframe\n",
    "df = pd.DataFrame()\n",
    "df['speech'] = paths\n",
    "df['label'] = labels\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:42:44.551432Z",
     "iopub.status.busy": "2022-07-19T10:42:44.551015Z",
     "iopub.status.idle": "2022-07-19T10:42:44.566541Z",
     "shell.execute_reply": "2022-07-19T10:42:44.565203Z",
     "shell.execute_reply.started": "2022-07-19T10:42:44.551402Z"
    }
   },
   "outputs": [],
   "source": [
    "df['label'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:42:58.725234Z",
     "iopub.status.busy": "2022-07-19T10:42:58.724724Z",
     "iopub.status.idle": "2022-07-19T10:42:58.998397Z",
     "shell.execute_reply": "2022-07-19T10:42:58.996901Z",
     "shell.execute_reply.started": "2022-07-19T10:42:58.725200Z"
    }
   },
   "outputs": [],
   "source": [
    "sns.countplot(df['label'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:43:07.172735Z",
     "iopub.status.busy": "2022-07-19T10:43:07.172344Z",
     "iopub.status.idle": "2022-07-19T10:43:07.183446Z",
     "shell.execute_reply": "2022-07-19T10:43:07.181571Z",
     "shell.execute_reply.started": "2022-07-19T10:43:07.172704Z"
    }
   },
   "outputs": [],
   "source": [
    "def waveshow(data, sr, emotion):\n",
    "    plt.figure(figsize=(10,4))\n",
    "    plt.title(emotion, size=20)\n",
    "    librosa.display.waveshow(data, sr=sr)\n",
    "    plt.show()\n",
    "    \n",
    "def spectogram(data, sr, emotion):\n",
    "    x = librosa.stft(data)\n",
    "    xdb = librosa.amplitude_to_db(abs(x))\n",
    "    plt.figure(figsize=(11,4))\n",
    "    plt.title(emotion, size=20)\n",
    "    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')\n",
    "    plt.colorbar()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:43:15.146018Z",
     "iopub.status.busy": "2022-07-19T10:43:15.145617Z",
     "iopub.status.idle": "2022-07-19T10:43:18.067666Z",
     "shell.execute_reply": "2022-07-19T10:43:18.065469Z",
     "shell.execute_reply.started": "2022-07-19T10:43:15.145988Z"
    }
   },
   "outputs": [],
   "source": [
    "emotion = 'fear'\n",
    "path = np.array(df['speech'][df['label']==emotion])[0]\n",
    "data, sampling_rate = librosa.load(path)\n",
    "waveshow(data, sampling_rate, emotion)\n",
    "spectogram(data, sampling_rate, emotion)\n",
    "Audio(path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:43:24.779447Z",
     "iopub.status.busy": "2022-07-19T10:43:24.779005Z",
     "iopub.status.idle": "2022-07-19T10:43:25.986353Z",
     "shell.execute_reply": "2022-07-19T10:43:25.984887Z",
     "shell.execute_reply.started": "2022-07-19T10:43:24.779418Z"
    }
   },
   "outputs": [],
   "source": [
    "emotion = 'angry'\n",
    "path = np.array(df['speech'][df['label']==emotion])[1]\n",
    "data, sampling_rate = librosa.load(path)\n",
    "waveshow(data, sampling_rate, emotion)\n",
    "spectogram(data, sampling_rate, emotion)\n",
    "Audio(path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:43:32.422303Z",
     "iopub.status.busy": "2022-07-19T10:43:32.421823Z",
     "iopub.status.idle": "2022-07-19T10:43:33.428438Z",
     "shell.execute_reply": "2022-07-19T10:43:33.426924Z",
     "shell.execute_reply.started": "2022-07-19T10:43:32.422269Z"
    }
   },
   "outputs": [],
   "source": [
    "emotion = 'disgust'\n",
    "path = np.array(df['speech'][df['label']==emotion])[0]\n",
    "data, sampling_rate = librosa.load(path)\n",
    "waveshow(data, sampling_rate, emotion)\n",
    "spectogram(data, sampling_rate, emotion)\n",
    "Audio(path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:43:41.213746Z",
     "iopub.status.busy": "2022-07-19T10:43:41.213339Z",
     "iopub.status.idle": "2022-07-19T10:43:42.342233Z",
     "shell.execute_reply": "2022-07-19T10:43:42.340910Z",
     "shell.execute_reply.started": "2022-07-19T10:43:41.213710Z"
    }
   },
   "outputs": [],
   "source": [
    "emotion = 'neutral'\n",
    "path = np.array(df['speech'][df['label']==emotion])[0]\n",
    "data, sampling_rate = librosa.load(path)\n",
    "waveshow(data, sampling_rate, emotion)\n",
    "spectogram(data, sampling_rate, emotion)\n",
    "Audio(path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:43:47.534600Z",
     "iopub.status.busy": "2022-07-19T10:43:47.533913Z",
     "iopub.status.idle": "2022-07-19T10:43:48.532577Z",
     "shell.execute_reply": "2022-07-19T10:43:48.530931Z",
     "shell.execute_reply.started": "2022-07-19T10:43:47.534566Z"
    }
   },
   "outputs": [],
   "source": [
    "emotion = 'sad'\n",
    "path = np.array(df['speech'][df['label']==emotion])[0]\n",
    "data, sampling_rate = librosa.load(path)\n",
    "waveshow(data, sampling_rate, emotion)\n",
    "spectogram(data, sampling_rate, emotion)\n",
    "Audio(path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:43:53.720595Z",
     "iopub.status.busy": "2022-07-19T10:43:53.720203Z",
     "iopub.status.idle": "2022-07-19T10:43:54.721637Z",
     "shell.execute_reply": "2022-07-19T10:43:54.720056Z",
     "shell.execute_reply.started": "2022-07-19T10:43:53.720565Z"
    }
   },
   "outputs": [],
   "source": [
    "emotion = 'ps'\n",
    "path = np.array(df['speech'][df['label']==emotion])[0]\n",
    "data, sampling_rate = librosa.load(path)\n",
    "waveshow(data, sampling_rate, emotion)\n",
    "spectogram(data, sampling_rate, emotion)\n",
    "Audio(path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:43:58.544511Z",
     "iopub.status.busy": "2022-07-19T10:43:58.544088Z",
     "iopub.status.idle": "2022-07-19T10:43:59.578960Z",
     "shell.execute_reply": "2022-07-19T10:43:59.577502Z",
     "shell.execute_reply.started": "2022-07-19T10:43:58.544481Z"
    }
   },
   "outputs": [],
   "source": [
    "emotion = 'happy'\n",
    "path = np.array(df['speech'][df['label']==emotion])[0]\n",
    "data, sampling_rate = librosa.load(path)\n",
    "waveshow(data, sampling_rate, emotion)\n",
    "spectogram(data, sampling_rate, emotion)\n",
    "Audio(path)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**FEATURE EXTRACTION**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:44:23.036399Z",
     "iopub.status.busy": "2022-07-19T10:44:23.035881Z",
     "iopub.status.idle": "2022-07-19T10:44:23.044740Z",
     "shell.execute_reply": "2022-07-19T10:44:23.043524Z",
     "shell.execute_reply.started": "2022-07-19T10:44:23.036366Z"
    }
   },
   "outputs": [],
   "source": [
    "def extract_mfcc(filename):\n",
    "    y, sr = librosa.load(filename, duration=3, offset=0.5)\n",
    "    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)\n",
    "    return mfcc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:44:29.590554Z",
     "iopub.status.busy": "2022-07-19T10:44:29.590158Z",
     "iopub.status.idle": "2022-07-19T10:44:29.778268Z",
     "shell.execute_reply": "2022-07-19T10:44:29.776623Z",
     "shell.execute_reply.started": "2022-07-19T10:44:29.590524Z"
    }
   },
   "outputs": [],
   "source": [
    "extract_mfcc(df['speech'][0])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T10:44:41.901665Z",
     "iopub.status.busy": "2022-07-19T10:44:41.901204Z",
     "iopub.status.idle": "2022-07-19T11:04:48.942853Z",
     "shell.execute_reply": "2022-07-19T11:04:48.940792Z",
     "shell.execute_reply.started": "2022-07-19T10:44:41.901633Z"
    }
   },
   "outputs": [],
   "source": [
    "X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T11:04:48.946566Z",
     "iopub.status.busy": "2022-07-19T11:04:48.946033Z",
     "iopub.status.idle": "2022-07-19T11:04:48.971028Z",
     "shell.execute_reply": "2022-07-19T11:04:48.969638Z",
     "shell.execute_reply.started": "2022-07-19T11:04:48.946503Z"
    }
   },
   "outputs": [],
   "source": [
    "X_mfcc\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T11:05:25.510139Z",
     "iopub.status.busy": "2022-07-19T11:05:25.509542Z",
     "iopub.status.idle": "2022-07-19T11:05:25.549206Z",
     "shell.execute_reply": "2022-07-19T11:05:25.547888Z",
     "shell.execute_reply.started": "2022-07-19T11:05:25.510092Z"
    }
   },
   "outputs": [],
   "source": [
    "X = [x for x in X_mfcc]\n",
    "X = np.array(X)\n",
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T11:05:32.400785Z",
     "iopub.status.busy": "2022-07-19T11:05:32.400158Z",
     "iopub.status.idle": "2022-07-19T11:05:32.413984Z",
     "shell.execute_reply": "2022-07-19T11:05:32.412118Z",
     "shell.execute_reply.started": "2022-07-19T11:05:32.400728Z"
    }
   },
   "outputs": [],
   "source": [
    "## input split\n",
    "X = np.expand_dims(X, -1)\n",
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T11:05:39.802854Z",
     "iopub.status.busy": "2022-07-19T11:05:39.802433Z",
     "iopub.status.idle": "2022-07-19T11:05:39.819186Z",
     "shell.execute_reply": "2022-07-19T11:05:39.817643Z",
     "shell.execute_reply.started": "2022-07-19T11:05:39.802802Z"
    }
   },
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import OneHotEncoder\n",
    "enc = OneHotEncoder()\n",
    "y = enc.fit_transform(df[['label']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T11:05:46.142413Z",
     "iopub.status.busy": "2022-07-19T11:05:46.141944Z",
     "iopub.status.idle": "2022-07-19T11:05:46.148993Z",
     "shell.execute_reply": "2022-07-19T11:05:46.147040Z",
     "shell.execute_reply.started": "2022-07-19T11:05:46.142368Z"
    }
   },
   "outputs": [],
   "source": [
    "y = y.toarray()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T11:06:07.420939Z",
     "iopub.status.busy": "2022-07-19T11:06:07.420503Z",
     "iopub.status.idle": "2022-07-19T11:06:07.430011Z",
     "shell.execute_reply": "2022-07-19T11:06:07.428505Z",
     "shell.execute_reply.started": "2022-07-19T11:06:07.420910Z"
    }
   },
   "outputs": [],
   "source": [
    "y.shape\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Create the LSTM Model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T11:06:43.142413Z",
     "iopub.status.busy": "2022-07-19T11:06:43.142000Z",
     "iopub.status.idle": "2022-07-19T11:06:52.477801Z",
     "shell.execute_reply": "2022-07-19T11:06:52.476518Z",
     "shell.execute_reply.started": "2022-07-19T11:06:43.142382Z"
    }
   },
   "outputs": [],
   "source": [
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, LSTM, Dropout\n",
    "\n",
    "model = Sequential([\n",
    "    LSTM(123, return_sequences=False, input_shape=(40,1)),\n",
    "    Dense(64, activation='relu'),\n",
    "    Dropout(0.2),\n",
    "    Dense(32, activation='relu'),\n",
    "    Dropout(0.2),\n",
    "    Dense(7, activation='softmax')\n",
    "])\n",
    "\n",
    "model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
    "model.summary()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T11:07:15.563663Z",
     "iopub.status.busy": "2022-07-19T11:07:15.563151Z",
     "iopub.status.idle": "2022-07-19T11:07:37.996042Z",
     "shell.execute_reply": "2022-07-19T11:07:37.994495Z",
     "shell.execute_reply.started": "2022-07-19T11:07:15.563561Z"
    }
   },
   "outputs": [],
   "source": [
    "# Train the model\n",
    "history = model.fit(X, y, validation_split=0.2, epochs=100, batch_size=512, shuffle=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Plot the results**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T11:08:03.031449Z",
     "iopub.status.busy": "2022-07-19T11:08:03.031038Z",
     "iopub.status.idle": "2022-07-19T11:08:03.260322Z",
     "shell.execute_reply": "2022-07-19T11:08:03.258900Z",
     "shell.execute_reply.started": "2022-07-19T11:08:03.031417Z"
    }
   },
   "outputs": [],
   "source": [
    "epochs = list(range(100))\n",
    "acc = history.history['accuracy']\n",
    "val_acc = history.history['val_accuracy']\n",
    "\n",
    "plt.plot(epochs, acc, label='train accuracy')\n",
    "plt.plot(epochs, val_acc, label='val accuracy')\n",
    "plt.xlabel('epochs')\n",
    "plt.ylabel('accuracy')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-07-19T11:08:18.252976Z",
     "iopub.status.busy": "2022-07-19T11:08:18.252562Z",
     "iopub.status.idle": "2022-07-19T11:08:18.492103Z",
     "shell.execute_reply": "2022-07-19T11:08:18.490863Z",
     "shell.execute_reply.started": "2022-07-19T11:08:18.252945Z"
    }
   },
   "outputs": [],
   "source": [
    "loss = history.history['loss']\n",
    "val_loss = history.history['val_loss']\n",
    "\n",
    "plt.plot(epochs, loss, label='train loss')\n",
    "plt.plot(epochs, val_loss, label='val loss')\n",
    "plt.xlabel('epochs')\n",
    "plt.ylabel('loss')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
