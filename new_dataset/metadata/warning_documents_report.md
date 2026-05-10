# Warning Documents Report

This table summarizes the documents involved in the parser and registry warnings reproduced from `new_dataset`.

| Issue | Document ID / Filename | Current Location | FRBR | Arabic Title | Note |
| --- | --- | --- | --- | --- | --- |
| FRBR fallback | `constitution_1988-11-03.xml` | `new_dataset/data/akn/constitution_1988-11-03.xml` | `/akn/dz/act/law/1988-11-03/[NUMBER]/main` | التعديل الدستوري المصادق عليه في إستفتاء 3 نوفمبر سنة 1988 | FRBR number is a placeholder, so the parser falls back to the filename stem. |
| FRBR fallback | `constitution_1989-02-23.xml` | `new_dataset/data/akn/constitution_1989-02-23.xml` | `/akn/dz/act/constitution/1989-02-23/main` | التعديل الدستوري المصادق عليه في إستفتاء 23 فبراير سنة 1989 | FRBR path shape is incomplete for the current parser, so it falls back to the filename stem. |
| Filename collision | `06-01_2006-02-20.xml` | `new_dataset/data/akn/06-01_2006-02-20.xml` | `/akn/dz/act/law/2005-02-06/05-01/main` | القانون المتعلق بالوقاية من تبييض الأموال وتمويل الإرهاب ومكافحتهما المعدل والمتمم | Filename implies `06-01_2006-02-20`, but internal FRBR resolves to `05-01_2005-02-06`. |
| Metadata mismatch | `12-2003_2012-11-28.xml` | `new_dataset/data/akn/12-2003_2012-11-28.xml` | `/akn/dz/act/law/2012-11-28/03-12/main` | النظام رقم 03-12 المتعلق بالوقاية من تبييض الأموال وتمويل الإرهاب ومكافحتهما | Filename suggests `12-2003`, while FRBR points to `03-12_2012-11-28`. |
| Metadata mismatch | `15-247_2015-09-16.xml` | `new_dataset/data/akn/15-247_2015-09-16.xml` | `/akn/dz/act/law/2015-09-16/15-247/main` | يتضمن تنظيم الصفقات العمومية وتفويضات المرفق العام | Registry warning is about `docNumber` saying presidential decree while the AKN root is `<act name="law">`. |
| Missing from corpus | `03-05_2003-07-19` | Not found in `new_dataset/data/akn` | `-` | أمر يتعلق بحقوق المؤلف والحقوق المجاورة | No AKN-linked file found in the dataset registry. |
| Missing from corpus | `03-10_2003-07-19` | No AKN XML found. Alternate file present: `new_dataset/data/pdf/قانون-03-10.pdf` | `-` | قانون حماية البيئة في إطار التنمية المستدامة | Present as a PDF filename only; no matching AKN XML was found under the canonical stem. |
| Missing from corpus | `06-15_2006-05-11` | Not found in `new_dataset/data/akn` | `-` | شروط وكيفيات تطبيق أحكام المادة 7 مكرر من قانون الأسرة | No AKN-linked file found in the dataset registry. |
| Missing from corpus | `11-04_2011-02-17` | Not found in `new_dataset/data/akn` | `-` | قانون يحدد القواعد المتعلقة بنشاط الترقية العقارية | No AKN-linked file found in the dataset registry. |
| Missing from corpus | `66-155_1966-06-08` | Not found in `new_dataset/data/akn` | `-` | أمر يتضمن قانون الإجراءات الجزائية (القديم - ملغى بقانون 25-14) | No AKN-linked file found in the dataset registry. Nearby `66-156_1966-06-08` is a different law number. |
