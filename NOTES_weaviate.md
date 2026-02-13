followed
https://docs.weaviate.io/deploy/installation-guides/docker-installation

when terminal closed and docker image stopped, when I restarted:
{"action":"startup","build_git_commit":"aa3ad25","build_go_version":"go1.25.7","build_image_tag":"v1.35.7","build_wv_version":"1.35.7","error":"could not join raft join list: context deadline exceeded. Weaviate detected this node to have state stored. If the DB is still loading up we will hit this timeout. You can try increasing/setting RAFT_BOOTSTRAP_TIMEOUT env variable to a higher value","level":"fatal","msg":"could not open cloud meta store","time":"2026-02-13T15:47:04Z"}

