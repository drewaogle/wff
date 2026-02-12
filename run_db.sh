docker run -p 8080:8080 -p 50051:50051 -e "PERSISTENCE_DATA_PATH=/var/lib/weaviate" -v ./data:/var/lib/weaviate cr.weaviate.io/semitechnologies/weaviate:1.35.7
