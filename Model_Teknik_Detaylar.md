# Model Teknik Detayları: Yöntemler ve Uygulama

Bu bölüm, önerilen Federe Kullanıcı ve Varlık Davranış Analizi (F-UEBA) çerçevesinin teknik yönlerini, sistem mimarisini, temel derin öğrenme modelini, anomali tespit mekanizmasını ve kullanılan iletişim verimliliği tekniklerini kapsamlı bir şekilde sunmaktadır.

## 1. F-UEBA Sistem Mimarisi

F-UEBA çerçevesi, federe öğrenme ortamında iletişim açısından verimli anomali tespitini kolaylaştırmak için tasarlanmıştır. Üç ana bileşenden oluşur:

*   **İstemciler (Veri Sahipleri):** Her istemci, kullanıcı davranış verilerinin yerel bir alt kümesini tutar. Yerel veri ön işleme, kendi özel verileri üzerinde model eğitimi ve model güncellemelerini (gradyanlar veya parametreler) merkezi sunucuya göndermekten sorumludurlar.
*   **Federasyon Sunucusu:** Merkezi sunucu, federe öğrenme sürecini yönetir. Birden fazla istemciden alınan model güncellemelerini toplar, küresel modeli günceller ve güncellenmiş küresel modeli istemcilere geri dağıtır. Sunucu, toplama için FedAvg (Federated Averaging) algoritmasını kullanır.
*   **Eklenti Mimarisi:** Çekirdek modeli veya FL mantığını değiştirmeden çeşitli iletişim verimliliği tekniklerini (örneğin, niceleme, seyreltme) uygulamak ve değerlendirmek için federe öğrenme hattına modüler bir eklenti mimarisi entegre edilmiştir.

Tüm çerçeve, Flower (flwr) federe öğrenme kütüphanesi kullanılarak uygulanmıştır; `ServerApp` ve `ClientApp` yapıları tanımlanmış, istemciler bir `NumPyClient` arayüzü aracılığıyla etkileşim kurmaktadır. Donanım hızlandırma, PyTorch'un CUDA desteği aracılığıyla sağlanmakta ve daha hızlı eğitim için NVIDIA GPU'ları kullanılmaktadır.

## 2. Veri Ön İşleme ve Özellik Mühendisliği

CERT Insider Threat r4.2 veri seti, gerçek dünya kullanıcı etkinliklerini simüle eden log verileriyle bu çalışmanın temelini oluşturmaktadır. Ön işleme ve özellik mühendisliği hattı, ham log verilerini derin öğrenme modeli için uygun bir formata dönüştürmek için kritik öneme sahiptir:

*   **Ham Veri Temizliği:** Ham log verilerinin başlangıç temizliği.
*   **Özellik Çıkarımı:** Le et al. (IEEE TNSM 2020, 2021) tarafından ilham alınan metodolojiyi takiben, temizlenmiş log verilerinden 508 adet kapsamlı davranışsal özellik çıkarılmıştır. Bu, Regex kullanılarak metin tabanlı boyut bilgilerinin sayısal değerlere dönüştürülmesini içerir.
*   **Normalizasyon:** Çıkarılan tüm özellikler `MinMaxScaler` kullanılarak 0-1 aralığına ölçeklendirilmiştir. Bu, farklı ölçeklere sahip özelliklerin modelin öğrenme sürecine eşit şekilde katkıda bulunmasını sağlar. `MinMaxScaler` nesnesi, eğitim ve çıkarım aşamaları arasında tutarlılık sağlamak için `pickle` kullanılarak kalıcı hale getirilmiştir.
*   **Kullanıcı Tanımlama:** Çalışan kimlikleri, model girdisi olarak hizmet etmek üzere `LabelEncoder` yöntemiyle sayısal olarak kodlanmıştır.
*   **Zamansal Bağlam (Kayar Pencere):** Kullanıcı etkinliklerinin zamansal bağlamını korumak için veriler kronolojik olarak sıralanmıştır. Veriler, 14 ardışık operasyonel adımdan (`window_size=14`) oluşan diziler halinde gruplandırılarak "Kayar Pencere" tekniği uygulanmıştır. Bu, modelin yalnızca anlık olayları değil, olaylar arasındaki zamansal ilişkileri de öğrenmesini sağlar.

## 3. Çekirdek Model: Çift Yönlü LSTM Otomatik Kodlayıcı

Normal kullanıcı davranış kalıplarını öğrenmek ve anormallikleri tespit etmek için derin öğrenme tabanlı **Çift Yönlü LSTM Otomatik Kodlayıcı** tasarlanmıştır. Modelin mimarisi ve yinelemeli iyileştirmeleri aşağıda detaylandırılmıştır:

*   **Temel Mimari:** Model, bir kodlayıcı-darboğaz-kod çözücü yapısından oluşur.
    *   **Kodlayıcı:** Kullanıcı davranışının sıkıştırılmış, gizli bir temsilini öğrenmek için girdi dizilerini işler.
    *   **Darboğaz:** Modelin yalnızca en belirgin davranışsal kalıpları yakalamasını sağlayan azaltılmış boyutlu bir katman (başlangıçta 192, 64 boyuta sıkılaştırılmıştır).
    *   **Kod Çözücü:** Gizli temsilden girdi dizisini yeniden yapılandırır.
*   **Mimari İyileştirmeler:**
    *   **Çift Yönlü LSTM:** Kodlayıcı, Çift Yönlü bir LSTM'ye yükseltilmiştir. Bu, modelin girdi dizilerini hem ileri hem de geri yönlerde işlemesine olanak tanır, bir dizi içindeki hem geçmiş hem de gelecek olaylardan daha zengin bağlamsal bilgileri yakalar ve "normal" davranış için daha sağlam bir temel oluşturur.
    *   **Katman Normalizasyonu:** Eğitimi stabilize etmek ve performansı artırmak için LSTM katmanları içinde uygulanmıştır.
    *   **Gürültü Giderme:** Otomatik kodlayıcı, girdileri yeniden yapılandırarak sağlam temsiller öğrenmek için örtük olarak tasarlanmıştır, bu da doğal olarak bir gürültü giderme etkisi sağlayabilir.
*   **Kayıp Fonksiyonu:** Girdi ile yeniden yapılandırılmış çıktı arasındaki farkı ölçen birincil kayıp fonksiyonu olarak Ortalama Kare Hata (MSE) kullanılmıştır. L1 Kaybı (MAE) ile deneyler yapılmış ancak kritik anomali sinyallerinin aşırı düzeltilmesi nedeniyle geri dönülmüştür.

## 4. Anomali Tespit Mekanizması

Anomali tespit mekanizması, LSTM Otomatik Kodlayıcının yeniden yapılandırma hatasını kullanır ve iç tehdit tespitinde yüksek hassasiyet elde etmek için çeşitli yinelemeli iyileştirmelerle geliştirilmiştir:

*   **Yeniden Yapılandırma Hatası:** Eğitim sırasında model, yalnızca normal kullanıcı davranışlarını yeniden yapılandırmayı öğrenir. Anormallikler, öğrenilen normal kalıplardan bir sapmayı gösteren yüksek bir yeniden yapılandırma hatasıyla tanımlanır.
*   **Z-Skor Tabanlı Anomali Puanlaması:**
    *   **Başlangıç Z-Skoru:** Her dizi için yeniden yapılandırmanın Ortalama Kare Hatası (MSE) hesaplanır. Anomali puanları, bu MSE'nin normal eğitim sırasında gözlemlenen MSE'lerin ortalaması ve standart sapmasıyla karşılaştırılmasıyla elde edilir.
    *   **Hibrit Z-Puanlaması:** Normal davranış ile anormallikler arasında daha keskin bir ayrım oluşturmak için hibrit bir yaklaşım tanıtılmıştır. Bir kullanıcının davranışı hem kendi geçmiş kalıplarına (`Yerel Z-Skoru`) hem de tüm kullanıcı popülasyonunun kalıplarına (`Küresel Z-Skoru`) göre karşılaştırılır. Bu, küresel olarak normal görünen ancak bir birey için önemli sapmalar olan "sinsi" tehditleri belirlemeye yardımcı olur.
    *   **En Üst K Özellik Anomali Odaklanması:** Yeniden yapılandırma hatasının 50'den fazla özellik arasında ortalaması yerine, sapma *her bir özellik için ayrı ayrı* hesaplanır. Belirli bir pencere için nihai tespit puanı, yalnızca *en anormal 5 özelliğe* odaklanır. Bu, iç tehditler genellikle hedeflendiği (örneğin, büyük dosya kopyalama ancak normal oturum açma süreleri) için saldırı sinyallerinin normal özellikler tarafından seyreltilmesini önler.
    *   **Zamansal Kalıcılık Puanlaması:** Kötü niyetli faaliyetler genellikle süreklidir. "Yanlış pozitif"leri ortadan kaldırmak için, kullanıcı düzeyindeki anomali puanı tek bir `max()` penceresine değil, *en kötü 3 pencerenin ortalamasına* dayanır. Bu, bir uyarıyı tetiklemek için sürekli bir sapma gerektirir ve geçici gürültüyü filtreler.
    *   **En Üst K Metriğinin Standardizasyonu:** En Üst K metriği, ham MSE kullanmak yerine kendi küresel dağılımına (eğitim sırasında hesaplanan) göre standartlaştırılır.
    *   **Çok Vektörlü Çeşitlilik Faktörü:** Bir "Çeşitlilik Artırma" faktörü tanıtılmıştır. Nihai anomali puanı, aynı anda kaç özelliğin saptığına bağlı olarak bir faktörle çarpılır. Bu, çok vektörlü tehditleri (örneğin, USB + Dosya + Mesai dışı etkinlik) tek vektörlü gürültüden üstel olarak ayırır, çünkü gerçek iç tehditler genellikle birden fazla davranışsal boyutta koordineli eylemleri içerir.
    *   **Kırpılmış İstatistikler:** Küresel temel çizgiler için %1 Kırpılmış Ortalama/Std kullanılır. Bu, gürültülü normal eğitim verilerinin en üst %1'ini göz ardı ederek "Normal" temel çizgisini daha temiz ve daha sağlam hale getirir.
*   **Eşikleme:** Anomali tespiti, nihai anomali puanının dinamik bir Z-Skor eşiği (örneğin, 3.5) ile karşılaştırılmasıyla gerçekleştirilir. İstenen hassasiyet ve geri çağırma hedeflerine ulaşmak için optimal eşiği belirlemek üzere duyarlılık analizi yapılır.

## 5. Federe Öğrenme Entegrasyonu ve İletişim Verimliliği

Federe öğrenme süreci, istemci katılımı ve model toplama için belirli konfigürasyonlarla Flower çerçevesi tarafından yönetilir:

*   **İstemci Bölümleme:** Veri seti dinamik olarak bölümlere ayrılır ve 50 istemci arasında dağıtılır.
*   **Toplama Algoritması:** Yerel model ağırlıklarını toplamak ve kolektif bir "Küresel Model" oluşturmak için sunucu tarafında FedAvg (Federated Averaging) algoritması kullanılır.
*   **Eğitim Turları:** Model, 50 iletişim turu boyunca eğitilir ve `fraction_fit=0.5` ile her turda istemcilerin %50'si katılır.
*   **Veri Dağıtım Senaryoları:** Sistemin sağlamlığını değerlendirmek için hem IID (Bağımsız ve Özdeş Dağıtılmış) hem de Non-IID (Dirichlet alpha=0.5 kullanılarak) veri dağıtım senaryoları değerlendirilir.
*   **İletişim Verimliliği Eklentileri:** FL'deki iletişim darboğazını ele almak için aşağıdaki gradyan sıkıştırma teknikleri eklenti olarak uygulanmıştır:
    *   **Niceleme (float32 -> float16):** Model parametrelerinin hassasiyetini 32 bitlik kayan noktadan 16 bitlik kayan noktaya düşürür, iletilen veri boyutunu önemli ölçüde azaltır.
    *   **En Üst K Seyreltme:** Yalnızca en büyük K gradyan (veya parametre) iletilir, bu da model güncellemelerini etkili bir şekilde seyreltir ve iletişim yükünü azaltır. Farklı seyreltme oranları (örneğin, 0.1, 0.05, 0.01) araştırılmıştır.
    *   **Kombine Teknikler:** Maksimum iletişim tasarrufu elde etmek için hem Niceleme hem de En Üst K Seyreltme birlikte uygulanır.

## 6. Değerlendirme Metrikleri

F-UEBA sisteminin performansı, hem tespit doğruluğunu hem de iletişim verimliliğini kapsayan kapsamlı bir metrik seti kullanılarak değerlendirilir:

*   **Tespit Doğruluğu:**
    *   **PR-AUC (Hassasiyet-Geri Çağırma Eğrisi Altındaki Alan):** Anomali tespitinde yaygın olarak bulunan dengesiz veri kümeleri için sağlam bir metrik.
    *   **Max-F1 Skoru:** Çeşitli eşikler arasında elde edilen maksimum F1 skoru.
    *   **Dengeli Doğruluk:** Her sınıfta elde edilen geri çağırmanın ortalaması.
    *   **Hassasiyet:** Tespit edilen tüm anormallikler arasında doğru şekilde tanımlanan anormalliklerin oranı.
    *   **Geri Çağırma:** Tüm gerçek anormallikler arasında doğru şekilde tanımlanan anormalliklerin oranı.
*   **İletişim Verimliliği:**
    *   **Toplam İletişim Maliyeti (MB):** Federe öğrenme süreci boyunca istemciler ve sunucu arasında aktarılan toplam veri miktarı.
    *   **Tur Başına Ortalama İletişim Maliyeti (MB):** İletişim turu başına ortalama veri aktarımı.
*   **Yakınsama:**
    *   **Yakınsama Turu:** Modelin PR-AUC'sinin zirve değerinin %95'ine ulaştığı ilk tur, kararlı öğrenmeyi gösterir.

Tüm deneylerde tekrarlanabilir sonuçlar elde etmek için tüm rastgeleleştiriciler için sabit bir seed kullanılır.