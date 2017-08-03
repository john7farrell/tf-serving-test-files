# Description: Tensorflow Serving.

package(
    default_visibility = ["//tensorflow_serving:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

# open source marker; do not remove

package_group(
    name = "internal",
    packages = [
        "//tensorflow_serving/...",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
)

py_binary(
    name = "cifar10_client",
    srcs = [
        "cifar10_client.py",
    ],
    deps = [
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
