for %%f in (object_detection/protos/*.proto) do (
protoc object_detection/protos/%%f --python_out=.
)