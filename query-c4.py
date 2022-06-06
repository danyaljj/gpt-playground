from opensearchpy import OpenSearch

client = OpenSearch(
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

query = {
    "size": 10000,
    "query": {
        "function_score": {
            "functions": [
                {
                    "random_score": {
                    }
                }
            ],
        }
    }
}
response = client.search(body=query, index="c4-en")

# dump in a json file
import json
fout = open("random_text_documents.json", "w")
fout.write(json.dumps(response, indent=4, sort_keys=True))
