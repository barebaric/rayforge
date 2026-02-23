# Отримання коду

Цей посібник охоплює, як отримати вихідний код Rayforge для розробки.

## Форкніть репозиторій

Зробіть форк [репозиторію Rayforge](https://github.com/barebaric/rayforge) на GitHub, щоб створити власну копію, де ви можете вносити зміни.

## Клонуйте ваш форк

```bash
git clone https://github.com/YOUR_USERNAME/rayforge.git
cd rayforge
```

## Додайте upstream репозиторій

Додайте оригінальний репозиторій як upstream remote щоб відстежувати зміни:

```bash
git remote add upstream https://github.com/barebaric/rayforge.git
```

## Перевірте репозиторій

Перевірте, що remotes налаштовані правильно:

```bash
git remote -v
```

Ви повинні побачити як ваш форк (origin), так і upstream репозиторій.

## Наступні кроки

Після отримання коду продовжіть з [Налаштування](setup) щоб сконфігурувати ваше середовище розробки.
