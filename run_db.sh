#!/bin/bash
ports=()
ports+=( -p 8080:8080 )
ports+=( -p 50051:50051 )
env=()
env+=( -e "RAFT_ENABLE_ONE_NODE_RECOVERY=true" )
env+=(-e "PERSISTENCE_DATA_PATH=/var/lib/weaviate" )
vols=()
vols+=( -v ./data:/var/lib/weaviate )

docker run -d ${ports[@]} ${env[@]} ${vols[@]} cr.weaviate.io/semitechnologies/weaviate:1.35.7
