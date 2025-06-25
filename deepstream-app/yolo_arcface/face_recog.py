import cv2
import numpy as np
import paddle
from paddle.inference import Config, create_predictor
import os
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import glob
import time

class Face_Recog:
    def __init__(self, model_dir, milvus_host='localhost', milvus_port='19530', collection_name='face_embeddings'):
        """
        Initialize the Face_Recog class with model and Milvus connection.

        Args:
            model_dir (str): Directory containing the Paddle inference model.
            milvus_host (str): Milvus server host (default: localhost).
            milvus_port (str): Milvus server port (default: 19530).
            collection_name (str): Name of the Milvus collection (default: face_embeddings).
        """
        self.model_dir = model_dir
        self.collection_name = collection_name
        self.dim = 512  # Embedding size of the model
        self.model = self.load_model_predict()
        self.connect_milvus(milvus_host, milvus_port)
        self.setup_milvus_collection()

    def connect_milvus(self, host, port):
        """Connect to the Milvus server."""
        try:
            connections.connect("default", host=host, port=port)
            print("Connected to Milvus")
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise

    def setup_milvus_collection(self):
        """Create or load the Milvus collection."""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            print(f"Loaded collection: {self.collection_name}")
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=255)
            ]
            schema = CollectionSchema(fields=fields, description="Face embeddings collection")
            self.collection = Collection(self.collection_name, schema)
            print(f"Created new collection: {self.collection_name}")

            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 100}
            }
            self.collection.create_index("embedding", index_params)
            print("Created index")

    def preprocess_image(self, input_data):
        """
        Preprocess an image for embedding extraction.

        Args:
            input_data (str or np.ndarray): Image path or image array.

        Returns:
            np.ndarray or None: Preprocessed image array or None if preprocessing fails.
        """
        if isinstance(input_data, str):
            img = cv2.imread(input_data)
            if img is None:
                print(f"Failed to read image: {input_data}")
                return None
        else:
            img = input_data

        if img is None or img.size == 0:
            print(f"Image is empty or invalid: {input_data if isinstance(input_data, str) else 'numpy array'}")
            return None

        try:
            img = cv2.resize(img, (112, 112))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = (img - 0.5) / 0.5
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)
            return img
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def load_model_predict(self):
        """Load the Paddle inference model."""
        infer_config = Config(
            f"{self.model_dir}/inference.pdmodel",
            f"{self.model_dir}/inference.pdiparams"
        )
        infer_config.enable_memory_optim()
        infer_config.enable_use_gpu(1000, 0)
        predictor = create_predictor(infer_config)
        return predictor

    def extract_embedding(self, image):
        """
        Extract embedding from an image.

        Args:
            image (str or np.ndarray): Image path or image array.

        Returns:
            np.ndarray or None: Flattened embedding or None if extraction fails.
        """
        input_data = self.preprocess_image(image)
        if input_data is None:
            return None

        input_names = self.model.get_input_names()
        input_handle = self.model.get_input_handle(input_names[0])
        input_handle.copy_from_cpu(input_data)
        self.model.run()
        output_names = self.model.get_output_names()
        output_handle = self.model.get_output_handle(output_names[0])
        embedding = output_handle.copy_to_cpu()
        return embedding.flatten()

    def add_face_db(self, label, image_path):
        """
        Add a single face embedding to Milvus.

        Args:
            label (str): Label for the face.
            image_path (str): Path to the face image.

        Returns:
            bool: True if insertion succeeds, False otherwise.
        """
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            print(f"Failed to read image: {image_path}")
            return False

        embedding = self.extract_embedding(img)
        if embedding is None:
            print(f"Failed to extract embedding for {image_path or label}")
            return False

        data = [
            [embedding],
            [label],
            [image_path or ""]
        ]
        self.collection.insert(data)
        print(f"Inserted embedding for {label}")
        self.collection.load()
        time.sleep(1)  # Wait for Milvus to sync
        print(f"Collection loaded, number of entities: {self.collection.num_entities}")
        return True

    def add_faces_from_directory(self, folder):
        """
        Add all face embeddings from a directory to Milvus.

        Args:
            folder (str): Directory containing face images.
        """
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder, ext)))

        if not image_files:
            print(f"No images found in {folder}")
            return

        embeddings = []
        labels = []
        image_paths = []

        for image_path in image_files:
            print(f"Processing image: {image_path}")
            image = cv2.imread(image_path)
            if image is None or image.size == 0:
                print(f"Failed to read image: {image_path}")
                continue

            embedding = self.extract_embedding(image)
            if embedding is None:
                print(f"Failed to extract embedding for {image_path}")
                continue

            label = os.path.splitext(os.path.basename(image_path))[0]
            embeddings.append(embedding)
            labels.append(label)
            image_paths.append(image_path)
            print(f"Prepared embedding for {label}")

        if embeddings:
            data = [embeddings, labels, image_paths]
            self.collection.insert(data)
            print(f"Inserted {len(embeddings)} embeddings into Milvus")
            self.collection.load()
            # Retry mechanism to ensure collection is loaded
            for _ in range(5):
                time.sleep(2)
                if self.collection.num_entities >= len(embeddings):
                    break
            print(f"Collection loaded, number of entities: {self.collection.num_entities}")
        else:
            print("No valid embeddings extracted")

    def match_face(self, frame_embedding, top_k=1, threshold=0.6):
        """
        Search for matching faces in Milvus.

        Args:
            frame_embedding (np.ndarray): Embedding of the input frame.
            top_k (int): Number of top matches to return.
            threshold (float): Similarity threshold for a match.

        Returns:
            tuple: (image_path, similarity, label) or (None, 0.0, "Unknown") if no match.
        """
        self.collection.load()
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        frame_embedding = frame_embedding / np.linalg.norm(frame_embedding)
        results = self.collection.search(
            data=[frame_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["label", "image_path"]
        )

        if not results:
            return None, 0.0, "Unknown"

        best_result = results[0][0]
        similarity = best_result.distance
        label = best_result.entity.get("label")
        image_path = best_result.entity.get("image_path")

        if similarity < threshold:
            return None, 0.0, "Unknown"

        return image_path, similarity, label

    def crop_object(self, image, obj_meta):
        """
        Crop an object and search for matching faces.

        Args:
            image (np.ndarray): Input image.
            obj_meta: Object metadata containing rectangle parameters.

        Returns:
            tuple: (cropped image, image_path, similarity, label).
        """
        rect_params = obj_meta.rect_params
        top = int(rect_params.top)
        left = int(rect_params.left)
        width = int(rect_params.width)
        height = int(rect_params.height)

        crop_img = image[top:top+height, left:left+width]
        embedding = self.extract_embedding(crop_img)
        if embedding is None:
            return None, None, 0.0, "Unknown"

        image_path, similarity, label = self.match_face(embedding)
        print(f"Best id: {image_path}")
        print(f"Best similarity: {similarity}")
        return crop_img, image_path, similarity, label

    def get_collection_fields(self):
        """Print the fields of the Milvus collection."""
        print("Collection fields:")
        for field in self.collection.schema.fields:
            print(f"{field.name} ({field.dtype})")

    def get_id_embedding(self, label, fields=["label", "image_path", "embedding"]):
        """
        Query embedding by label from the face_embeddings collection.

        Args:
            label (str): Label of the face to query.
            fields (list): Fields to retrieve (default: label, image_path, embedding).

        Returns:
            dict or None: Dictionary with queried fields or None if no match.
        """
        self.collection.load()
        expr = f"label == '{label}'"
        results = self.collection.query(
            expr=expr,
            output_fields=fields,
            limit=1
        )

        if results:
            result = results[0]
            result["embedding"] = np.array(result["embedding"], dtype=np.float32)
            return result
        else:
            print(f"No embedding found for label: {label}")
            return None

    def delete_face_db(self, label=None, image_path=None, delete_all=False):
        """
        Delete face embeddings from the Milvus collection.

        Args:
            label (str, optional): Label of the face to delete.
            image_path (str, optional): Image path of the face to delete.
            delete_all (bool): If True, delete the entire collection.

        Returns:
            bool: True if deletion succeeds, False otherwise.
        """
        self.collection.load()
        
        if delete_all:
            utility.drop_collection(self.collection_name)
            print(f"Dropped entire collection: {self.collection_name}")
            self.setup_milvus_collection()
            return True

        if not label and not image_path:
            print("Must provide label or image_path to delete specific data")
            return False

        expr = []
        if label:
            expr.append(f"label == '{label}'")
        if image_path:
            expr.append(f"image_path == '{image_path}'")
        expr = " && ".join(expr)

        results = self.collection.query(expr=expr, output_fields=["id", "label", "image_path"])
        if not results:
            print(f"No data found with condition: {expr}")
            return False

        self.collection.delete(expr)
        print(f"Deleted {len(results)} entities with condition: {expr}")
        return True

    def list_stored_faces(self, limit=100):
        """
        List stored faces in the Milvus collection.

        Args:
            limit (int): Maximum number of records to return.

        Returns:
            list: List of stored face records.
        """
        self.collection.load()
        print(f"Loading collection for query, current entities: {self.collection.num_entities}")
        results = self.collection.query(
            expr="id >= 0",
            output_fields=["id", "label", "image_path"],
            limit=limit
        )
        print(f"\n=== List of faces in collection ({len(results)} records) ===")
        for result in results:
            print(f"ID: {result['id']}, Label: {result['label']}, Image Path: {result['image_path']}")
        return results

# if __name__ == "__main__":
#     face_recog = Face_Recog('/path/to/arcface_iresnet50_v1.0_infer')
#     # Print initial DB state
#     print("\n=== Initial state ===")
#     face_recog.list_stored_faces(limit=100)
    
#     # Add all images from directory
#     print("\n=== Adding images from directory ===")
#     face_recog.add_faces_from_directory('/path/to/saved_faces')
    
# #     # Add a single image
#     face_recog.add_face_db("add_frame", "/path/to/out_crops/stream_0/frame_3.jpg")
#     # Print DB state after adding
#     print("\n=== State after adding ===")
#     face_recog.list_stored_faces(limit=100)
    
#     embed = face_recog.get_id_embedding('add_frame')
#     print(embed)
#     # Delete entire DB
#     print("\n=== Deleting entire collection ===")
#     delete_db = face_recog.delete_face_db(delete_all=True)
    
#     # Print state after deletion
#     print("\n=== State after deletion ===")
#     face_recog.list_stored_faces(limit=100)