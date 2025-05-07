from typing import List, Optional

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlmodel import Session, select

from models.track import AudioFeature, Track


class AudioVisualizer:
    """Class for creating visualization of audio features"""

    def __init__(self, session: Session):
        self.session = session

    def get_tracks_dataframe(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve track with their audio features as a Dataframe"""
        statement = select(Track, AudioFeature).join(
            AudioFeature, Track.id == AudioFeature.track_id
        )

        if limit:
            statement = statement.limit(limit)

        results = self.session.exec(statement).all()

        data = []
        for track, feature in results:
            track_data = {
                "id": track.id,
                "title": track.title,
                "artist": track.artist,
                "duration": track.duration,
                "file_path": track.file_path,
            }

            for attr_name in dir(feature):
                if attr_name.startswith("_") or callable(getattr(feature, attr_name)):
                    continue

                value = getattr(feature, attr_name)
                if isinstance(value, (int, float, str, bool)) and not isinstance(
                    value, dict
                ):
                    track_data[attr_name] = value

            data.append(track_data)

        return pd.DataFrame(data)

    def feature_distribution(self, feature_name: str) -> go.Figure:
        """Create a histogram showing distribution of specific feature"""
        df = self.get_tracks_dataframe()

        if feature_name not in df.columns:
            raise ValueError(f"Feature '{feature_name}' not found in data")

        fig = px.histogram(
            df,
            x=feature_name,
            color="artist",
            hover_data=["title", "artist", "duration"],
            title=f"Distribution of {feature_name}",
        )

        fig.update_layout(xaxis_title=feature_name, yaxis_tile="Count", height=600)

        return fig

    def feature_scatter(self, x_feature: str, y_feature: str) -> go.Figure:
        """Create a scatter plot comparing two features"""

        df = self.get_tracks_dataframe()

        for feature in [x_feature, y_feature]:
            if feature not in df.columns:
                raise ValueError(f"Feature '{feature}' not found in data")

        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            color="artist",
            size="duration",
            hover_data=["title", "artist"],
            title=f"{x_feature} vs {y_feature}",
        )

        fig.update_layout(height=600, xaxis_title=x_feature, yaxis_title=y_feature)

        return fig

    def artist_feature_radar(
        self, artists: List[str], features: List[str]
    ) -> go.Figure:
        """Create a radar chart comparing features across artists"""

        df = self.get_tracks_dataframe()

        for feature in features:
            if feature not in df.columns:
                raise ValueError(f"Feature '{feature}' not found in data")

        artist_features = df.groupby("artists")[features].mean().reset_index()

        if artists:
            artist_features = artist_features[artist_features["artist"].isin(artists)]

        for feature in features:
            min_val = artist_features[feature].min()
            max_val = artist_features[feature].max()
            if max_val > min_val:
                artist_features[features] = (artist_features[feature] - min_val) / (
                    max_val - min_val
                )

        fig = go.Figure()

        for _, row in artist_features.iterrows():
            fig.add_trace(
                go.Scatterpolar(
                    r=row[features].values,
                    theta=features,
                    fill="toself",
                    name=row["artist"],
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Artist Feature Comparison",
            height=600,
        )

        return fig

    def tempo_distribution_by_artist(self) -> go.Figure:
        """Create a violin plot showing tempo distribution by artist"""
        df = self.get_tracks_dataframe()

        fig = px.violin(
            df,
            y="artist",
            x="tempo",
            color="artist",
            box=True,
            points="all",
            hover_data=["title", "duration"],
            title="Tempo Distribution by Artist",
        )

        fig.update_layout(
            xaxis_title="Tempo (BPM)",
            yaxis_title="Artist",
            height=600,
            showlegend=False,
        )

        return fig

    def feature_correlation_heatmap(self, features: List[str] = None) -> go.Figure:
        df = self.get_tracks_dataframe()

        if not features:
            features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
            features = [f for f in features if f != "id" and f != "track_id"]

        corr_matrix = df[features].corr()

        fig = px.imshow(
            corr_matrix,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Heatmap",
        )

        fig.update_layout(height=800, width=800)

        return fig

    def feature_similarity_network(self, threshold: float = 0.7) -> go.Figure:
        """Create a network graph of features based on correlation"""
        df = self.get_tracks_dataframe()

        # Get numeric features
        features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        features = [f for f in features if f != "id" and f != "track_id"]

        # Calculate correlation matrix
        corr_matrix = df[features].corr().abs()

        # Create network edges where correlation exceeds threshold
        edges = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if corr_matrix.iloc[i, j] >= threshold:
                    edges.append((features[i], features[j], corr_matrix.iloc[i, j]))

        # Create graph using networkx
        G = nx.Graph()

        # Add nodes
        for feature in features:
            G.add_node(feature)

        # Add edges
        for source, target, weight in edges:
            G.add_edge(source, target, weight=weight)

        # Calculate layout
        pos = nx.spring_layout(G)

        # Create edges trace
        edge_x = []
        edge_y = []
        edge_text = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            weight = G.edges[edge]["weight"]
            edge_text.append(f"Correlation: {weight:.2f}")

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        # Create nodes trace
        node_x = []
        node_y = []
        node_text = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                size=10,
                colorbar=dict(
                    thickness=15,
                    title="Node Connections",
                    xanchor="left",
                    titleside="right",
                ),
                line_width=2,
            ),
        )

        # Color nodes by number of connections
        node_adjacencies = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))

        node_trace.marker.color = node_adjacencies

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Feature Correlation Network",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800,
            ),
        )

        return fig

    def track_spectrogram(self, track_id: int) -> Optional[go.Figure]:
        """Create a spectrogram visualization for a track"""
        # This requires waveform data to be pre-computed
        track = self.session.get(Track, track_id)
        if not track or not hasattr(track, "waveform") or not track.waveform:
            return None

        waveform = track.waveform

        if not waveform.spectrogram_data:
            return None

        # Create figure
        fig = go.Figure(
            data=go.Heatmap(
                z=waveform.spectrogram_data["spectrogram"],
                x=waveform.spectrogram_data["times"],
                y=waveform.spectrogram_data["frequencies"],
                colorscale="Viridis",
                colorbar=dict(title="Energy"),
            )
        )

        fig.update_layout(
            title=f"Spectrogram: {track.title} - {track.artist}",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            yaxis_type="log",
            height=600,
        )

        return fig

    def track_waveform(self, track_id: int) -> Optional[go.Figure]:
        """Create a waveform visualization for a track"""
        track = self.session.get(Track, track_id)
        if not track or not hasattr(track, "waveform") or not track.waveform:
            return None

        waveform = track.waveform

        # Create figure
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=waveform.waveform_data["times"],
                y=waveform.waveform_data["amplitudes"],
                mode="lines",
                line=dict(color="rgb(31, 119, 180)", width=1),
                name="Waveform",
            )
        )

        fig.update_layout(
            title=f"Waveform: {track.title} - {track.artist}",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=400,
        )

        return fig

    def track_feature_dashboard(self, track_id: int) -> go.Figure:
        """Create a comprehensive dashboard for a single track"""
        track = self.session.get(Track, track_id)
        if not track or not track.audio_feature:
            return None

        # Create subplot figure
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Waveform",
                "Spectral Features",
                "Rhythmic Features",
                "Tonal Features",
                "Feature Comparison",
                "Chroma Features",
            ),
            specs=[
                [{"type": "xy"}, {"type": "polar"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "bar"}, {"type": "heatmap"}],
            ],
        )

        # Add waveform if available
        if hasattr(track, "waveform") and track.waveform:
            fig.add_trace(
                go.Scatter(
                    x=track.waveform.waveform_data["times"],
                    y=track.waveform.waveform_data["amplitudes"],
                    mode="lines",
                    name="Waveform",
                ),
                row=1,
                col=1,
            )

        # Add radar chart of spectral features
        feature = track.audio_feature
        spectral_features = [
            "spectral_centroid_mean",
            "spectral_bandwidth_mean",
            "spectral_contrast_mean",
            "spectral_rolloff_mean",
            "zero_crossing_rate_mean",
        ]

        # Normalize to 0-1 for visualization
        df = self.get_tracks_dataframe()
        normalized_values = []
        for f in spectral_features:
            min_val = df[f].min()
            max_val = df[f].max()
            val = getattr(feature, f)
            if max_val > min_val:
                normalized_values.append((val - min_val) / (max_val - min_val))
            else:
                normalized_values.append(0)

        fig.add_trace(
            go.Scatterpolar(
                r=normalized_values,
                theta=spectral_features,
                fill="toself",
                name="Spectral Features",
            ),
            row=1,
            col=2,
        )

        # Add rhythmic features (tempo, beats, onsets)
        fig.add_trace(
            go.Bar(
                x=["tempo", "num_beats", "num_onsets"],
                y=[feature.tempo, feature.num_beats, feature.num_onsets],
                name="Rhythmic Features",
            ),
            row=2,
            col=1,
        )

        # Add tonal features (key, key confidence)
        feature_vals = [
            feature.zero_crossing_rate_mean,
            feature.rms_mean,
            feature.tempo / 200.0,  # Normalize tempo
        ]
        fig.add_trace(
            go.Bar(
                x=["Zero Crossing", "RMS Energy", "Tempo (norm)"],
                y=feature_vals,
                name="Tonal & Energy",
            ),
            row=2,
            col=2,
        )

        # Feature comparison with dataset averages
        avg_vals = df[["spectral_centroid_mean", "rms_mean", "tempo"]].mean()
        track_vals = [feature.spectral_centroid_mean, feature.rms_mean, feature.tempo]

        fig.add_trace(
            go.Bar(x=["Brightness", "Energy", "Tempo"], y=track_vals, name="Track"),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=["Brightness", "Energy", "Tempo"],
                y=[
                    avg_vals["spectral_centroid_mean"],
                    avg_vals["rms_mean"],
                    avg_vals["tempo"],
                ],
                name="Average",
            ),
            row=3,
            col=1,
        )

        # Add chroma features if available in raw features
        if track.features and "chroma_mean" in track.features:
            chroma_labels = [
                "C",
                "C#",
                "D",
                "D#",
                "E",
                "F",
                "F#",
                "G",
                "G#",
                "A",
                "A#",
                "B",
            ]
            chroma_values = track.features["chroma_mean"]

            fig.add_trace(
                go.Heatmap(
                    z=[chroma_values],
                    x=chroma_labels,
                    y=["Chroma"],
                    colorscale="Viridis",
                ),
                row=3,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title=f"Audio Analysis: {track.title} - {track.artist}",
            height=1000,
            showlegend=True,
        )

        return fig
