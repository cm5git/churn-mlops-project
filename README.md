Customer Churn Prediction: from notebook to a tested, containerised service on Azure
**Try it live:** [churn-api.bluewater-bf71659a.uksouth.azurecontainerapps.io/docs](https://churn-api.bluewater-bf71659a.uksouth.azurecontainerapps.io/docs). Open **POST /predict**, click **Try it out**, then **Execute**. It scales to zero when idle, so the first request may take a few seconds. Temporary: it runs on an Azure trial subscription.

A churn model for the public Telco customer dataset, taken from exploratory analysis all the way to a REST API that is tested, containerised and deployed to Azure Container Apps by a GitHub Actions pipeline.

The model is deliberately simple. The point of the project is the engineering around it: avoiding train/serve skew, validating inputs, comparing models fairly, and shipping the exact image that was tested.
How it fits together
              git push to main

                     |

                     v

        +---------------------------+

        |  GitHub Actions           |

        |  1. pytest (55 tests)     |

        |  2. docker build          |

        |  3. smoke test container  |

        |  4. push image  ----------+--->  Azure Container Registry

        |  5. update app  ----------+--->  Azure Container Apps --> public HTTPS API

        |  6. smoke test live URL   |

        +---------------------------+

Pull requests run steps 1 to 3 only. They never touch Azure.
Results
7,043 customers, 26.5% churned. Stratified 80/20 split (5,634 train, 1,409 test), 18 input fields. Two models share identical preprocessing and class weighting:



Champion: logistic regression
Challenger: gradient boosting
ROC-AUC, 5-fold CV on training data
0.845 ± 0.012
0.845 ± 0.011
PR-AUC, 5-fold CV on training data
0.659 ± 0.019
0.660 ± 0.019
ROC-AUC, test set
0.839
0.842
PR-AUC, test set
0.629
0.658
Churn recall / precision at a 0.5 cutoff
0.78 / 0.51
0.80 / 0.53
Model file
9 KB
289 KB


The two models tie. The cross-validated scores differ by far less than their fold-to-fold spread, and the larger test-set PR-AUC gap is the size of difference a test set with only 374 churners can produce by chance. I kept the simpler model: it is interpretable, 30 times smaller and easier to explain.
Why recall over precision. Both models use class_weight="balanced", so they are pushed to catch churners. On the test set the champion catches 290 of 374 churners (recall 0.78) at the cost of 281 false alarms (precision 0.51). That trade is deliberate: a missed churner costs more than a wasted retention offer.
Why not accuracy. The champion's accuracy is 74%, barely above the 73.5% you would get by predicting "no churn" for everyone, which is why the table reports recall and threshold-free AUCs instead.

Figures come from one run on the Kaggle CSV and can differ slightly across library versions.
Project structure
.

├── app.py                     FastAPI service: /health and /predict

├── src/

│   ├── features.py            shared pipeline: preprocessing + champion/challenger models

│   └── train.py               training, cross-validation, MLflow logging, model export

├── model/

│   ├── model.joblib           champion (served by default)

│   └── challenger.joblib

├── tests/                     55 tests

│   ├── test_api.py

│   ├── test_features.py

│   └── test_models.py

├── scripts/smoke_test.py      end-to-end check against any running instance

├── notebooks/01_exploration.ipynb

├── Dockerfile

├── requirements.txt           serving dependencies, pinned

├── requirements-dev.txt       tests, MLflow, notebooks

└── .github/workflows/ci_cd.yml
Run it locally
git clone https://github.com/cm5git/churn-mlops-project.git

cd churn-mlops-project

python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements-dev.txt

Download the Telco Customer Churn dataset and save it as data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv (the data/ folder is gitignored).

# Train. -m runs it as part of the src package.

python3 -m src.train                       # champion: logistic regression

python3 -m src.train --model challenger    # challenger: gradient boosting

# Compare the runs side by side

mlflow ui --backend-store-uri sqlite:///mlflow.db

# Serve the API, then open http://127.0.0.1:8000/docs

uvicorn app:app --reload

MODEL_PATH=model/challenger.joblib uvicorn app:app    # serve the challenger instead

# Test

pytest tests/ -v

python3 scripts/smoke_test.py http://localhost:8000

With Docker:

docker build -t churn-api .

docker run -p 8000:8000 churn-api
API
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{

  "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",

  "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",

  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",

  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",

  "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",

  "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5

}'

{"churn_probability": 0.8413, "churn_prediction": 1}

Categorical fields accept only the values seen in training. A typo such as "Contract": "Two years" returns a 422 that lists the allowed values, rather than being silently treated as a different contract type.
Testing
55 tests run on every push, with no dataset, trained model or MLflow server required:

API: endpoints, the decision threshold including its boundary, input validation, model loading, and an integration test that runs the real API with a real fitted pipeline.
Features: one row must be encoded exactly like the same row in a batch; baseline categories; "No internet service" treated as "No"; unseen categories; column order.
Models: both models behave the same from the outside, the challenger can never overwrite the champion, and the evaluation helpers work.
Container smoke test (scripts/smoke_test.py): health, predictions, high-risk scores above low-risk, changing only the contract changes the prediction, and a typo is rejected. It runs against the container in CI and again against the live deployment.
CI/CD
.github/workflows/ci_cd.yml runs two jobs. test runs the suite. docker, which starts only if test passes, builds the image once, smoke tests it as a running container, and, on pushes to main only, pushes that same image to Azure Container Registry, updates the Container App, and smoke tests the live URL.

Traceable and reversible. Images are tagged with the commit hash. Rolling back means pointing the app at an earlier tag.
No stored credentials. GitHub authenticates to Azure with OpenID Connect. The three repository secrets (AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_SUBSCRIPTION_ID) are identifiers, not passwords. Azure trusts only workflows running from main of this repository.
Least privilege. The deployment identity can push to one registry and manage one resource group. The running app has a separate identity that can only pull images.

Azure setup (one-off) Resources: a resource group, a Basic Container Registry, a Container Apps environment, one Container App that scales to zero (0.5 vCPU, 1 GiB, at most one replica), and two managed identities. Run in Azure Cloud Shell:

RG=churn-rg

LOCATION=uksouth

ACR_NAME=<globally-unique-name>

az provider register --namespace Microsoft.App --wait

az provider register --namespace Microsoft.OperationalInsights --wait

az provider register --namespace Microsoft.ContainerRegistry --wait

az group create --name $RG --location $LOCATION

az acr create --resource-group $RG --name $ACR_NAME --sku Basic

# Identity GitHub uses to push images and update the app (passwordless)

az identity create --resource-group $RG --name churn-github-id

GH_PRINCIPAL_ID=$(az identity show -g $RG -n churn-github-id --query principalId -o tsv)

az identity federated-credential create --name github-main --identity-name churn-github-id \

  --resource-group $RG --issuer https://token.actions.githubusercontent.com \

  --subject "repo:<owner>@<owner_id>/<repo>@<repo_id>:ref:refs/heads/main" \

  --audiences api://AzureADTokenExchange

ACR_ID=$(az acr show --name $ACR_NAME --query id -o tsv)

RG_ID=$(az group show --name $RG --query id -o tsv)

az role assignment create --assignee-object-id $GH_PRINCIPAL_ID --assignee-principal-type ServicePrincipal --role AcrPush --scope $ACR_ID

az role assignment create --assignee-object-id $GH_PRINCIPAL_ID --assignee-principal-type ServicePrincipal --role Contributor --scope $RG_ID

# Identity the running app uses to pull its image

az identity create --resource-group $RG --name churn-pull-id

PULL_ID=$(az identity show -g $RG -n churn-pull-id --query id -o tsv)

PULL_PRINCIPAL=$(az identity show -g $RG -n churn-pull-id --query principalId -o tsv)

az role assignment create --assignee-object-id $PULL_PRINCIPAL --assignee-principal-type ServicePrincipal --role AcrPull --scope $ACR_ID

# After CI has pushed a first image, create the environment and the app

az extension add --name containerapp --upgrade --yes

az containerapp env create --name churn-env --resource-group $RG --location $LOCATION

az containerapp create --name churn-api --resource-group $RG --environment churn-env \

  --image $ACR_NAME.azurecr.io/churn-api:<first-commit-hash> \

  --user-assigned $PULL_ID --registry-server $ACR_NAME.azurecr.io --registry-identity $PULL_ID \

  --target-port 8000 --ingress external --min-replicas 0 --max-replicas 1 --cpu 0.5 --memory 1.0Gi

Then add the three repository secrets and set ACR_NAME and IMAGE in the workflow. The federated credential's subject must match what GitHub sends exactly. Since mid-2026 new repositories include the numeric owner and repository IDs in it; if login fails with AADSTS700213, the error message shows the exact subject to register.

To remove everything: az group delete --name churn-rg --yes --no-wait.
Design decisions
One scikit-learn Pipeline holds the encoding, scaling and classifier, saved as a single artifact. The API passes raw fields straight in, so there is no second copy of the preprocessing to drift out of sync with training.
Category merging inside the encoder. "No internet service" and "No phone service" are treated as "No" by declaring only ["No", "Yes"] as known values, so the saved pipeline uses only built-in scikit-learn parts and loads anywhere scikit-learn is installed.
Strict request validation with Literal and Field types, so bad input is rejected before it reaches the model.
Lazy model loading from a configurable path. The app starts and answers /health without a model present, and tests can swap the model out.
A lean serving image: eight pinned packages and no MLflow, Jupyter or test tools, running as a non-root user.
Build once, ship what was tested. The image that passes the smoke test is the image that gets deployed.
Fair model comparison: identical preprocessing and class weights, threshold-free metrics, and cross-validation on the training data only, leaving the test set for a final check.
What went wrong, and what I learned
A train/serve skew bug. My first API repeated the training preprocessing by hand, using pd.get_dummies(..., drop_first=True). On a full dataset that is correct. On a single customer, every categorical column has one value, so drop_first dropped all of them and the model silently saw every customer as a month-to-month customer with no services. Predictions still looked plausible, and my own sanity check (a high-risk and a low-risk customer scored very differently) hid the problem because tenure and charges alone separated them. I found it while writing tests, fixed it by moving everything into one pipeline, and added regression tests, including one that changes only the contract and checks that the prediction moves.

A login failure from a changed token format. The first GitHub-to-Azure login failed because GitHub's token subject now includes immutable numeric owner and repository IDs, which my trust rule didn't. The error message printed the subject that was actually sent. I replaced the rule with an exact match, which is also the safer one because names can be recycled and IDs cannot.
Limitations and next steps
Public, unauthenticated endpoint, capped at one replica. A real service needs authentication, rate limiting and request logging.
Public sample data. The results say nothing about real churn.
No monitoring or feedback loop. There are no ground-truth labels in production, so live accuracy can't be measured, and drift detection is not implemented. The plan would be to log requests without personal data, compare their distributions with the training data using a library such as Evidently, and track outcomes as labels arrive.
The challenger isn't deployed. Switching is one setting (MODEL_PATH), and Container Apps can split traffic between revisions for a canary rollout, but I haven't exercised that.
A fixed 0.5 cutoff. The right threshold should come from the cost of a retention offer against the value of a retained customer.
Allowed values live in two places (the API and the training data) and should be generated from one source.
The model file is committed to git, which is fine at 9 KB. Larger models belong in a model registry.
Deployed on a trial subscription, so any live URL is temporary.
Data
IBM's Telco Customer Churn sample dataset, as published on Kaggle.
