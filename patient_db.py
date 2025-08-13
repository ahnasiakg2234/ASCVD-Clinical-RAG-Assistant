# patient_db.py

import sqlite3
from typing import List, Optional

DB_NAME = "clinical_patients.db"


def create_connection():
    """Creates and returns a database connection."""
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    """Initializes the database and creates tables if they don't exist."""
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                age INTEGER CHECK(age BETWEEN 40 AND 79),
                sex TEXT CHECK(sex IN ('Male', 'Female')),
                race TEXT,
                total_cholesterol INTEGER,
                hdl_cholesterol INTEGER,
                systolic_bp INTEGER,
                on_hypertension_treatment INTEGER CHECK(on_hypertension_treatment IN (0, 1)),
                smoker INTEGER CHECK(smoker IN (0, 1)),
                diabetic INTEGER CHECK(diabetic IN (0, 1)),
                family_history TEXT,
                elevated_ldl TEXT,
                ckd TEXT,
                metabolic_syndrome TEXT,
                inflammation TEXT,
                elevated_triglycerides TEXT,
                premature_menopause TEXT,
                ethnicity_risk TEXT
            )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id    INTEGER NOT NULL,
            timestamp     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            risk_percent  REAL NOT NULL,
            report        TEXT NOT NULL,
            FOREIGN KEY(patient_id) REFERENCES patients(id) ON DELETE CASCADE
        )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS current_medications (
                med_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INT REFERENCES patients(id),
                drug_name TEXT,
                dose TEXT,
                frequency TEXT,
                route TEXT,
                start_date DATE,
                indication TEXT
            )
        ''')
        conn.commit()


class Patient:
    """Represents a patient record in the database."""

    def __init__(self, first_name: str, last_name: str, age: int, sex: str, race: str,
                 total_cholesterol: int, hdl_cholesterol: int, systolic_bp: int,
                 on_hypertension_treatment: int, smoker: int, diabetic: int,
                 family_history: str, elevated_ldl: str, ckd: str, metabolic_syndrome: str,
                 inflammation: str, elevated_triglycerides: str, premature_menopause: str,
                 ethnicity_risk: str, id: Optional[int] = None):
        self.id = id
        self.first_name = first_name
        self.last_name = last_name
        self.age = age
        self.sex = sex
        self.race = race
        self.total_cholesterol = total_cholesterol
        self.hdl_cholesterol = hdl_cholesterol
        self.systolic_bp = systolic_bp
        self.on_hypertension_treatment = on_hypertension_treatment
        self.smoker = smoker
        self.diabetic = diabetic
        self.family_history = family_history
        self.elevated_ldl = elevated_ldl
        self.ckd = ckd
        self.metabolic_syndrome = metabolic_syndrome
        self.inflammation = inflammation
        self.elevated_triglycerides = elevated_triglycerides
        self.premature_menopause = premature_menopause
        self.ethnicity_risk = ethnicity_risk

    def __repr__(self):
        return (f"Patient(id={self.id}, name={self.first_name} {self.last_name}, age={self.age}, "
                f"sex={self.sex}, race={self.race})")

    def save(self):
        """Inserts or updates the patient record in the database."""
        with create_connection() as conn:
            cursor = conn.cursor()
            if self.id is None:
                cursor.execute('''
                    INSERT INTO patients (
                        first_name, last_name, age, sex, race,
                        total_cholesterol, hdl_cholesterol, systolic_bp,
                        on_hypertension_treatment, smoker, diabetic,
                        family_history, elevated_ldl, ckd, metabolic_syndrome,
                        inflammation, elevated_triglycerides, premature_menopause,
                        ethnicity_risk
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (self.first_name, self.last_name, self.age, self.sex, self.race,
                      self.total_cholesterol, self.hdl_cholesterol, self.systolic_bp,
                      self.on_hypertension_treatment, self.smoker, self.diabetic,
                      self.family_history, self.elevated_ldl, self.ckd, self.metabolic_syndrome,
                      self.inflammation, self.elevated_triglycerides, self.premature_menopause,
                      self.ethnicity_risk))
                self.id = cursor.lastrowid
            else:
                cursor.execute('''
                    UPDATE patients SET
                        first_name = ?, last_name = ?, age = ?, sex = ?, race = ?,
                        total_cholesterol = ?, hdl_cholesterol = ?, systolic_bp = ?,
                        on_hypertension_treatment = ?, smoker = ?, diabetic = ?,
                        family_history = ?, elevated_ldl = ?, ckd = ?, metabolic_syndrome = ?,
                        inflammation = ?, elevated_triglycerides = ?, premature_menopause = ?,
                        ethnicity_risk = ?
                    WHERE id = ?
                ''', (self.first_name, self.last_name, self.age, self.sex, self.race,
                      self.total_cholesterol, self.hdl_cholesterol, self.systolic_bp,
                      self.on_hypertension_treatment, self.smoker, self.diabetic,
                      self.family_history, self.elevated_ldl, self.ckd, self.metabolic_syndrome,
                      self.inflammation, self.elevated_triglycerides, self.premature_menopause,
                      self.ethnicity_risk, self.id))
            conn.commit()

    def delete(self):
        """Deletes the patient record and resequences all patient IDs."""
        if self.id is not None:
            with create_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM patients WHERE id = ?', (self.id,))
                conn.commit()
            
            resequence_patient_ids()
            self.id = None

    @classmethod
    def get(cls, patient_id: int) -> Optional['Patient']:
        """Retrieves a patient by their ID."""
        with create_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
            row = cursor.fetchone()
            if row:
                return cls(**dict(row))
        return None

    class Medication:
        def __init__(self, patient_id: int, drug_name: str, dose: str, frequency: str,
                     route: Optional[str] = None, start_date: Optional[str] = None,
                     indication: Optional[str] = None, med_id: Optional[int] = None):
            self.med_id = med_id
            self.patient_id = patient_id
            self.drug_name = drug_name
            self.dose = dose
            self.frequency = frequency
            self.route = route
            self.start_date = start_date
            self.indication = indication

        def save(self):
            """Inserts or updates the medication record."""
            with create_connection() as conn:
                cursor = conn.cursor()
                if self.med_id is None:
                    cursor.execute('''
                                   INSERT INTO current_medications (patient_id, drug_name, dose, frequency, route,
                                                                    start_date, indication)
                                   VALUES (?, ?, ?, ?, ?, ?, ?)
                                   ''', (self.patient_id, self.drug_name, self.dose, self.frequency,
                                         self.route, self.start_date, self.indication))
                    self.med_id = cursor.lastrowid
                else:
                    cursor.execute('''
                                   UPDATE current_medications
                                   SET drug_name  = ?,
                                       dose       = ?,
                                       frequency  = ?,
                                       route      = ?,
                                       start_date = ?,
                                       indication = ?
                                   WHERE med_id = ?
                                   ''', (self.drug_name, self.dose, self.frequency, self.route,
                                         self.start_date, self.indication, self.med_id))
                conn.commit()

        def delete(self):
            """Deletes this medication."""
            if self.med_id is not None:
                with create_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM current_medications WHERE med_id = ?', (self.med_id,))
                    conn.commit()
                    self.med_id = None

        @classmethod
        def get_by_patient(cls, patient_id: int) -> List['Medication']:
            with create_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM current_medications WHERE patient_id = ?', (patient_id,))
                rows = cursor.fetchall()
                return [cls(**dict(row)) for row in rows]

        def __repr__(self):
            return f"{self.drug_name} {self.dose} ({self.frequency})"

    @classmethod
    def get_all(cls) -> List['Patient']:
        """Retrieves all patients from the database."""
        with create_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM patients')
            rows = cursor.fetchall()
            return [cls(**dict(row)) for row in rows]


class Medication:
    def __init__(self, patient_id: int, drug_name: str, dose: str, frequency: str,
                 route: Optional[str] = None, start_date: Optional[str] = None,
                 indication: Optional[str] = None, med_id: Optional[int] = None):
        self.med_id = med_id
        self.patient_id = patient_id
        self.drug_name = drug_name
        self.dose = dose
        self.frequency = frequency
        self.route = route
        self.start_date = start_date
        self.indication = indication

    def save(self):
        """Inserts or updates the medication record."""
        with create_connection() as conn:
            cursor = conn.cursor()
            if self.med_id is None:
                cursor.execute('''
                               INSERT INTO current_medications (patient_id, drug_name, dose, frequency, route,
                                                                start_date, indication)
                               VALUES (?, ?, ?, ?, ?, ?, ?)
                               ''', (self.patient_id, self.drug_name, self.dose, self.frequency,
                                     self.route, self.start_date, self.indication))
                self.med_id = cursor.lastrowid
            else:
                cursor.execute('''
                               UPDATE current_medications
                               SET drug_name  = ?,
                                   dose       = ?,
                                   frequency  = ?,
                                   route      = ?,
                                   start_date = ?,
                                   indication = ?
                               WHERE med_id = ?
                               ''', (self.drug_name, self.dose, self.frequency, self.route,
                                     self.start_date, self.indication, self.med_id))
            conn.commit()

    def delete(self):
        """Deletes this medication."""
        if self.med_id is not None:
            with create_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM current_medications WHERE med_id = ?', (self.med_id,))
                conn.commit()
                self.med_id = None

    @classmethod
    def get_by_patient(cls, patient_id: int) -> List['Medication']:
        with create_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM current_medications WHERE patient_id = ?', (patient_id,))
            rows = cursor.fetchall()
            return [cls(**dict(row)) for row in rows]

    def __repr__(self):
        return f"{self.drug_name} {self.dose} ({self.frequency})"

def clear_all_patients():
    """Deletes all patient records from the database."""
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM patients')
        conn.commit()
        print("All patient records cleared.")


def resequence_patient_ids():
    """Resequences patient IDs to be contiguous, starting from 1."""
    with create_connection() as conn:
        cursor = conn.cursor()
        try:
            # Step 1: Disable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = OFF")

            # Step 2: Create a temporary table with the same structure
            cursor.execute('''
                CREATE TABLE patients_temp (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    age INTEGER,
                    sex TEXT,
                    race TEXT,
                    total_cholesterol INTEGER,
                    hdl_cholesterol INTEGER,
                    systolic_bp INTEGER,
                    on_hypertension_treatment INTEGER,
                    smoker INTEGER,
                    diabetic INTEGER,
                    family_history TEXT,
                    elevated_ldl TEXT,
                    ckd TEXT,
                    metabolic_syndrome TEXT,
                    inflammation TEXT,
                    elevated_triglycerides TEXT,
                    premature_menopause TEXT,
                    ethnicity_risk TEXT
                )
            ''')

            # Step 3: Copy data from the old table to the new one
            cursor.execute('''
                INSERT INTO patients_temp (
                    first_name, last_name, age, sex, race,
                    total_cholesterol, hdl_cholesterol, systolic_bp,
                    on_hypertension_treatment, smoker, diabetic,
                    family_history, elevated_ldl, ckd, metabolic_syndrome,
                    inflammation, elevated_triglycerides, premature_menopause,
                    ethnicity_risk
                )
                SELECT
                    first_name, last_name, age, sex, race,
                    total_cholesterol, hdl_cholesterol, systolic_bp,
                    on_hypertension_treatment, smoker, diabetic,
                    family_history, elevated_ldl, ckd, metabolic_syndrome,
                    inflammation, elevated_triglycerides, premature_menopause,
                    ethnicity_risk
                FROM patients ORDER BY id
            ''')

            # Step 4: Drop the old table
            cursor.execute('DROP TABLE patients')

            # Step 5: Rename the new table to the original name
            cursor.execute('ALTER TABLE patients_temp RENAME TO patients')

            # Step 6: Commit the transaction
            conn.commit()

        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            conn.rollback()
        finally:
            # Step 7: Re-enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")


def recreate_assessments_table():
    """Re-creates the assessments table with ON DELETE CASCADE foreign key."""
    with create_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = OFF")
            cursor.execute("DROP TABLE IF EXISTS assessments")
            cursor.execute('''
                CREATE TABLE assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    risk_percent REAL NOT NULL,
                    report TEXT NOT NULL,
                    FOREIGN KEY(patient_id) REFERENCES patients(id) ON DELETE CASCADE
                )
            ''')
            conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred while recreating the assessments table: {e}")
            conn.rollback()
        finally:
            cursor.execute("PRAGMA foreign_keys = ON")


if __name__ == "__main__":
    initialize_db()
    recreate_assessments_table()  # Ensure the assessments table is correctly set up
    # clear_all_patients()  # Uncomment to clear all patient records

    # Example of creating a new patient
    if not Patient.get_all():
        print("No patients found, creating a new one.")
        patient1 = Patient(
            first_name='John', last_name='Doe', age=55, sex='Male', race='White/Other',
            total_cholesterol=200, hdl_cholesterol=50, systolic_bp=120,
            on_hypertension_treatment=0, smoker=0, diabetic=0,
            family_history="Unknown", elevated_ldl="Unknown", ckd="Unknown",
            metabolic_syndrome="Unknown", inflammation="Unknown",
            elevated_triglycerides="Unknown", premature_menopause="Unknown",
            ethnicity_risk="Unknown"
        )
        patient1.save()
        print(f"Saved: {patient1}")

    # Get all patients
    all_patients = Patient.get_all()
    print("\nAll patients:")
    for p in all_patients:
        print(p)

    if all_patients:
        # Get a single patient
        first_patient_id = all_patients[0].id
        retrieved_patient = Patient.get(first_patient_id)
        print(f"\nRetrieved: {retrieved_patient}")

        # Update a patient
        if retrieved_patient:
            retrieved_patient.age = 56
            retrieved_patient.save()
            print(f"\nUpdated: {Patient.get(retrieved_patient.id)}")

        # Delete a patient
        # if retrieved_patient:
        #     retrieved_patient.delete()
        #     print(f"\nDeleted patient. Patirace inent exists: {Patient.get(first_patient_id) is not None}")

    patient = Patient.get(1)
    if patient:
        new_med = Medication(
            patient_id=patient.id,
            drug_name='Atorvastatin',
            dose='20mg',
            frequency='daily',
            route='oral',
            start_date='2025-01-01',
            indication='Hyperlipidemia'
        )
        new_med.save()

        meds = Medication.get_by_patient(patient.id)
        print(f"Medications for {patient.first_name} {patient.last_name}:")
        for med in meds:
            print(med)


def resequence_patient_ids():
    """Resequences patient IDs to be contiguous, starting from 1."""
    with create_connection() as conn:
        cursor = conn.cursor()
        try:
            # Step 1: Disable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = OFF")

            # Step 2: Create a temporary table with the same structure
            cursor.execute('''
                CREATE TABLE patients_temp (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    age INTEGER,
                    sex TEXT,
                    race TEXT,
                    total_cholesterol INTEGER,
                    hdl_cholesterol INTEGER,
                    systolic_bp INTEGER,
                    on_hypertension_treatment INTEGER,
                    smoker INTEGER,
                    diabetic INTEGER,
                    family_history TEXT,
                    elevated_ldl TEXT,
                    ckd TEXT,
                    metabolic_syndrome TEXT,
                    inflammation TEXT,
                    elevated_triglycerides TEXT,
                    premature_menopause TEXT,
                    ethnicity_risk TEXT
                )
            ''')

            # Step 3: Copy data from the old table to the new one
            cursor.execute('''
                INSERT INTO patients_temp (
                    first_name, last_name, age, sex, race,
                    total_cholesterol, hdl_cholesterol, systolic_bp,
                    on_hypertension_treatment, smoker, diabetic,
                    family_history, elevated_ldl, ckd, metabolic_syndrome,
                    inflammation, elevated_triglycerides, premature_menopause,
                    ethnicity_risk
                )
                SELECT
                    first_name, last_name, age, sex, race,
                    total_cholesterol, hdl_cholesterol, systolic_bp,
                    on_hypertension_treatment, smoker, diabetic,
                    family_history, elevated_ldl, ckd, metabolic_syndrome,
                    inflammation, elevated_triglycerides, premature_menopause,
                    ethnicity_risk
                FROM patients ORDER BY id
            ''')

            # Step 4: Drop the old table
            cursor.execute('DROP TABLE patients')

            # Step 5: Rename the new table to the original name
            cursor.execute('ALTER TABLE patients_temp RENAME TO patients')

            # Step 6: Commit the transaction
            conn.commit()

        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            conn.rollback()
        finally:
            # Step 7: Re-enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")


def recreate_assessments_table():
    """Re-creates the assessments table with ON DELETE CASCADE foreign key."""
    with create_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = OFF")
            cursor.execute("DROP TABLE IF EXISTS assessments")
            cursor.execute('''
                CREATE TABLE assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    risk_percent REAL NOT NULL,
                    report TEXT NOT NULL,
                    FOREIGN KEY(patient_id) REFERENCES patients(id) ON DELETE CASCADE
                )
            ''')
            conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred while recreating the assessments table: {e}")
            conn.rollback()
        finally:
            cursor.execute("PRAGMA foreign_keys = ON")




if __name__ == "__main__":
    initialize_db()
    recreate_assessments_table()  # Ensure the assessments table is correctly set up
    # clear_all_patients()  # Uncomment to clear all patient records

    # Example of creating a new patient
    patient = Patient.get(1)
    if patient:
        new_med = Medication(
            patient_id=patient.id,
            drug_name='Atorvastatin',
            dose='20mg',
            frequency='daily',
            route='oral',
            start_date='2025-01-01',
            indication='Hyperlipidemia'
        )
        new_med.save()

        meds = Medication.get_by_patient(patient.id)
        print(f"Medications for {patient.first_name} {patient.last_name}:")
        for med in meds:
            print(med)
    if not Patient.get_all():
        print("No patients found, creating a new one.")
        patient1 = Patient(
            first_name='John', last_name='Doe', age=55, sex='Male', race='White/Other',
            total_cholesterol=200, hdl_cholesterol=50, systolic_bp=120,
            on_hypertension_treatment=0, smoker=0, diabetic=0,
            family_history="Unknown", elevated_ldl="Unknown", ckd="Unknown",
            metabolic_syndrome="Unknown", inflammation="Unknown",
            elevated_triglycerides="Unknown", premature_menopause="Unknown",
            ethnicity_risk="Unknown"
        )
        patient1.save()
        print(f"Saved: {patient1}")

    # Get all patients
    all_patients = Patient.get_all()
    print("\nAll patients:")
    for p in all_patients:
        print(p)

    if all_patients:
        # Get a single patient
        first_patient_id = all_patients[0].id
        retrieved_patient = Patient.get(first_patient_id)
        print(f"\nRetrieved: {retrieved_patient}")

        # Update a patient
        if retrieved_patient:
            retrieved_patient.age = 56
            retrieved_patient.save()
            print(f"\nUpdated: {Patient.get(retrieved_patient.id)}")

        # Delete a patient
        # if retrieved_patient:
        #     retrieved_patient.delete()
        #     print(f"\nDeleted patient. Patirace inent exists: {Patient.get(first_patient_id) is not None}")