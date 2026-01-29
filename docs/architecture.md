# Architektura Systemu (Pipeline)

Projekt opiera się na sekwencyjnym przetwarzaniu klatek wideo (Video Processing Pipeline). Poniższy diagram przedstawia przepływ danych od surowego pliku wejściowego do finalnego dashboardu.

## Schemat Przetwarzania Danych

```mermaid
graph TD
    subgraph Input
    A[Input Video .mp4] 
    end

    subgraph "Core AI Engine"
    A --> B(YOLO Object Detection);
    B --> C(ByteTrack Tracking);
    C --> D{Object Class?};
    D -->|Player| E(Team Color Assignment);
    D -->|Ball| F(Ball Interpolation);
    
    A --> G(Camera Movement Estimation);
    end

    subgraph "Geometry & Physics"
    E & F & G --> H(View Transformer);
    H -->|Perspective Transform| I[2D Metric Coordinates];
    I --> J(Speed & Distance Calculation);
    end

    subgraph Output
    J --> K[Output CSV Data];
    K --> L[Streamlit Dashboard];
    end
    
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
```